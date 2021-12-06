import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from ..utils import MultiplicativeCrossAttention
from torch.nn import functional as F


class LSTMEncoder(nn.Module):

    def __init__(self,tokenizer,embedding_dim,hidden_size):
        super().__init__()
        vocab = tokenizer.vocab
        num_embeddings = len(vocab)
        self.pad_idx = vocab[tokenizer.pad_token]
        self.hidden_size = hidden_size

        self.tokenizer = tokenizer
        self.emb = nn.Embedding(num_embeddings,embedding_dim,padding_idx=self.pad_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,hidden_size=hidden_size,
            num_layers=1,bias=True,batch_first=True,dropout=0.,
            bidirectional=True,proj_size=0
        )
        self.proj_h = nn.Linear(2*hidden_size,hidden_size,bias=False)
        self.proj_c = nn.Linear(2*hidden_size,hidden_size,bias=False)

    def forward(self,src_input_ids,src_attention_mask):
        embeddings_seq = self.emb(src_input_ids)
        packed_embeddings = pack_padded_sequence(embeddings_seq,src_attention_mask.sum(dim=1).cpu().type(torch.int),batch_first=True)
        out_seq, (h_enc, c_enc) = self.lstm(packed_embeddings)
        out_seq, _ = pad_packed_sequence(out_seq,batch_first=True,padding_value=self.pad_idx)
        h_enc_final = h_enc.transpose(0,1).reshape(-1,2*self.hidden_size)
        c_enc_final = c_enc.transpose(0,1).reshape(-1,2*self.hidden_size)
        h_dec_0 = self.proj_h(h_enc_final)
        c_dec_0 = self.proj_h(c_enc_final)

        return out_seq, (h_dec_0, c_dec_0)

        
class LSTMDecoder(nn.Module):

    def __init__(self,tokenizer,embedding_dim,hidden_size,dropout=0.0):
        super().__init__()
        vocab = tokenizer.vocab
        num_embeddings = len(vocab)
        self.pad_idx = vocab[tokenizer.pad_token]
        self.hidden_size = hidden_size

        self.tokenizer = tokenizer
        self.emb = nn.Embedding(num_embeddings,embedding_dim,padding_idx=self.pad_idx)
        self.lstm = nn.LSTMCell(
            input_size=embedding_dim+hidden_size,hidden_size=hidden_size,bias=True
        )
        self.attention = MultiplicativeCrossAttention(hidden_size,2*hidden_size)
        self.linear_dec = nn.Linear(3*hidden_size,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear_vocab = nn.Linear(hidden_size,num_embeddings)

    def forward(self,enc_out,src_attention_mask,tgt_input_ids,h_dec_0,c_dec_0):
        
        # La entrada del decoder tiene desde <s> hasta el anterior a </s>
        batch_size = tgt_input_ids.size(0)
        seq_len = tgt_input_ids.size(1) - 1
        embeddings_seq = self.emb(tgt_input_ids[:,:-1])
        
        # Inicializo los vectores para el cálculo en cada paso
        o_prev = torch.zeros(batch_size,self.hidden_size,dtype=torch.float,device=enc_out.device)
        h_prev, c_prev = h_dec_0, c_dec_0

        o_list = []
        for i in range(seq_len):
            y = embeddings_seq[:,i,:]
            # y_bar = torch.cat((y,o_prev),dim=-1)
            # h_prev, c_prev = self.lstm(y_bar,(h_prev,c_prev))
            # a = self.attention(h_prev,enc_out,src_attention_mask)
            # u = torch.cat((a,h_prev),dim=-1)
            # v = self.linear_dec(u)
            # o_prev = self.dropout(torch.tanh(v))
            h_prev, c_prev, o_prev = self.step(enc_out,src_attention_mask,y,o_prev,h_prev,c_prev)
            o_list.append(o_prev)
        
        o = torch.cat(o_list,dim=1).view(-1,seq_len,self.hidden_size)
        P = F.log_softmax(self.linear_vocab(o),dim=-1)

        return P

    def step(self,enc_out,src_attention_mask,y,o_prev,h_prev,c_prev):
        y_bar = torch.cat((y,o_prev),dim=-1)
        h_new, c_new = self.lstm(y_bar,(h_prev,c_prev))
        a = self.attention(h_new,enc_out,src_attention_mask)
        u = torch.cat((a,h_new),dim=-1)
        v = self.linear_dec(u)
        o_new = self.dropout(torch.tanh(v))
        return h_new, c_new, o_new
        

        
class LSTMEncoderDecoder(nn.Module):

    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder.cuda(0)
        self.decoder = decoder.cuda(1)
        
    def forward(self,src_encoded_input,tgt_encoded_input):

        src_input_ids = src_encoded_input['input_ids'].cuda(0)
        src_attention_mask = src_encoded_input['attention_mask'].cuda(0)
        src_sent_enc, (h_dec_0, c_dec_0) = self.encoder(src_input_ids,src_attention_mask)

        src_sent_enc = src_sent_enc.cuda(1)
        h_dec_0 = h_dec_0.cuda(1)
        c_dec_0 = c_dec_0.cuda(1)
        src_attention_mask = src_encoded_input['attention_mask'].cuda(1)

        tgt_input_ids = tgt_encoded_input['input_ids'].cuda(1)
        tgt_attention_mask = tgt_encoded_input['attention_mask'].cuda(1)
        probs = self.decoder(src_sent_enc,src_attention_mask,tgt_input_ids,h_dec_0, c_dec_0)

        # La salida del decoder se compara con la secuencia que va
        # desde el siguiente a <s> hasta </s>
        tgt_gold_words_log_probs = torch.gather(probs,index=tgt_input_ids[:,1:].unsqueeze(-1),dim=-1).squeeze(-1) * tgt_attention_mask[:,1:]
        loss = -tgt_gold_words_log_probs.sum() 
        return probs, loss


    def greedy_decode(self,src_sent,max_len):

        src_encoded_input = self.encoder.tokenizer(pd.Series([src_sent]))
        src_input_ids = src_encoded_input['input_ids'].cuda(0)
        src_attention_mask = src_encoded_input['attention_mask'].cuda(0)
        enc_out, (h_dec_0, c_dec_0) = self.encoder(src_input_ids,src_attention_mask)
        src_attention_mask = src_attention_mask.cuda(1)
        enc_out = enc_out.cuda(1)
        h_dec_0 = h_dec_0.cuda(1)
        c_dec_0 = c_dec_0.cuda(1)

        start_id = self.decoder.tokenizer.vocab[self.decoder.tokenizer.start_token]
        end_id = self.decoder.tokenizer.vocab[self.decoder.tokenizer.end_token]
        pad_id = self.decoder.tokenizer.vocab[self.decoder.tokenizer.pad_token]

        # La entrada del decoder tiene desde <s> hasta el anterior a </s>
        curr_token_id = torch.tensor([[start_id]],dtype=torch.long,device=enc_out.device)
        pred_sent = [start_id]
        curr_token_id_val = start_id
        
        # Inicializo los vectores para el cálculo en cada paso
        o_prev = torch.zeros(1,self.decoder.hidden_size,dtype=torch.float,device=enc_out.device)
        h_prev, c_prev = h_dec_0, c_dec_0

        for _ in range(max_len):
            if curr_token_id_val == end_id:
                pred_sent.append(pad_id)
                continue

            y = self.decoder.emb(curr_token_id).squeeze(1)
            h_prev, c_prev, o_prev = self.decoder.step(enc_out,src_attention_mask,y,o_prev,h_prev,c_prev)
            curr_token_id = F.log_softmax(self.decoder.linear_vocab(o_prev),dim=-1).max(dim=-1).indices
            curr_token_id_val = curr_token_id.item()
            pred_sent.append(curr_token_id_val)
            
        return pred_sent


def init_lstm_model(
            src_tokenizer,
            tgt_tokenizer,
            embedding_dim,
            hidden_size,
            dropout=0.0
    ):
    encoder = LSTMEncoder(src_tokenizer,embedding_dim,hidden_size)
    decoder = LSTMDecoder(tgt_tokenizer,embedding_dim,hidden_size,dropout)
    model = LSTMEncoderDecoder(encoder,decoder)
    return model