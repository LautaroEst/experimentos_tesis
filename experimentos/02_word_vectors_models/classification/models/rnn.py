import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from .embedding import ELMOEmbedding


class RNNClassifier(nn.Module):

    def __init__(self,embedding_layer,rnn,bidirectional,hidden_size,num_outs,num_layers,dropout):
        super().__init__()
        self.emb = embedding_layer
        
        if rnn == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_layer.embedding_dim,hidden_size=hidden_size,
                    num_layers=num_layers,bias=True,batch_first=True,
                    dropout=0,bidirectional=bidirectional)
        else:
            raise NameError("RNN type not supported")

        self.rnn_type = rnn
        self.dropout = nn.Dropout(dropout)
        self.in_linear = 2*hidden_size if bidirectional else hidden_size
        self.linear_out = nn.Linear(self.in_linear,num_outs)


    def forward(self,batch_sentences):
        batch_embeddings, attention_mask = self.emb(batch_sentences)
        # device = next(self.parameters()).device
        # batch_embeddings = batch_embeddings.to(device=device)
        seq_len = attention_mask.cpu().sum(dim=1).long()
        
        packed_seq = pack_padded_sequence(batch_embeddings,seq_len,batch_first=True,enforce_sorted=False)
        out, hidden = self.rnn(packed_seq)
        hidden = self.dropout(hidden[0].transpose(0,1)[:,-2:,:].reshape(-1,self.in_linear))
        # if self.rnn_type == 'LSTM':
        #     hidden = hidden[0]
        scores = self.linear_out(hidden)
        #out = pad_packed_sequence(out,batch_first=True,padding_value=0)
        return scores

    def to(self,*args,**kwargs):
        super().to(*args,**kwargs)

        if isinstance(self.emb,ELMOEmbedding):
            device = None
            for arg in args:
                if isinstance(arg,torch.device):
                    device = arg
                    break
            if device is None:
                for arg in kwargs.values():
                    if isinstance(arg,torch.device):
                        device = arg
                        break
            self.emb.device = device