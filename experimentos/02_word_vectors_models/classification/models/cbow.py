import torch
from torch import nn


class CBOWClassifier(nn.Module):


    def __init__(self,embedding_layer,embedding_dim,hidden_sizes,num_outs,dropout):
        super().__init__()
        self.emb = embedding_layer
        self.linear_modules = nn.ModuleList([
            nn.Linear(embedding_dim,hidden_sizes[0]) 
        ] + [
            nn.Linear(hidden_sizes[i],hidden_sizes[i+1]) for i in range(0,len(hidden_sizes)-1)
        ])
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Linear(hidden_sizes[-1],num_outs)


    def forward(self,batch_sentences):
        batch_embeddings, attention_mask = self.emb(batch_sentences)
        device = next(self.parameters()).device
        batch_embeddings = batch_embeddings.to(device=device)
        attention_mask = attention_mask.to(device=device)
        seq_len = attention_mask.sum(dim=1,keepdims=True)
        x = batch_embeddings.sum(dim=1) / seq_len
        x = torch.relu(x)
        for m in self.linear_modules:
            x = m(x)
            x = torch.relu(x)
            x = self.dropout(x)
        scores = self.out_linear(x)
        return scores


