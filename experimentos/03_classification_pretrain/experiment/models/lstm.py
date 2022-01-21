import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from ..utils.tokenizers import WordTokenizer
import math



class LSTMModel(nn.Module):

    def __init__(self,hidden_size,embedding_dim,hidden_layers,num_embeddings,pad_idx,dropout,num_outs):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings,embedding_dim,pad_idx)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=hidden_layers,
            bias=True,
            batch_first=True,
            dropout=dropout if hidden_layers > 1 else 0.,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.in_linear = 2 * hidden_size
        self.linear_out = nn.Linear(self.in_linear,num_outs)

    def forward(self,input_ids,attention_mask):
        batch_embeddings = self.emb(input_ids)
        seq_len = attention_mask.cpu().sum(dim=1).long()
        
        packed_seq = pack_padded_sequence(batch_embeddings,seq_len,batch_first=True,enforce_sorted=False)
        _, hidden = self.rnn(packed_seq)
        hidden = self.dropout(hidden[0].transpose(0,1)[:,-2:,:].reshape(-1,self.in_linear))
        scores = self.linear_out(hidden)
        return scores


def init_lstm_model(hidden_size,embedding_dim,hidden_layers,num_embeddings,pad_idx,dropout,num_outs):
    return LSTMModel(
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        hidden_layers=hidden_layers,
        num_embeddings=num_embeddings,
        pad_idx=pad_idx,
        dropout=dropout,
        num_outs=num_outs
    )


def load_lstm_model_from_checkpoint(
        hidden_size,
        embedding_dim,
        hidden_layers,
        tokenizer,
        checkpoint,
        dropout,
        num_outs
    ):
    model = init_lstm_model(hidden_size,embedding_dim,hidden_layers,len(tokenizer.vocab),tokenizer.vocab[tokenizer.pad_token],dropout,num_outs)
    melisa_tokenizer = WordTokenizer.from_json(os.path.join(checkpoint,"tokenizer_config.json"))
    state_dict = torch.load(os.path.join(checkpoint,"best_model_checkpoint.pkl"))["model_state_dict"]
    emb_weight = state_dict["emb.weight"]
    state_dict["emb.weight"] = torch.randn(len(tokenizer.vocab),embedding_dim) / math.sqrt(embedding_dim)
    model.load_state_dict(state_dict)
    count = 0
    for tk, idx in tokenizer.vocab.items():
        if tk in melisa_tokenizer.vocab.keys():
            count += 1
            model.emb.weight.data[idx,:] = emb_weight.data[melisa_tokenizer.vocab[tk],:]
    print("Found {} embeddings in pretraining".format(count))
    return model