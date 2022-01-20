import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence



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