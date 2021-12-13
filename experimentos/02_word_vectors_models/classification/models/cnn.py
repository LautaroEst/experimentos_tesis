import torch
from torch import nn
from .embedding import ELMOEmbedding


class CNNClassifier(nn.Module):
    def __init__(self,embedding_layer,filter_sizes,n_filters,nclasses,dropout):
        super().__init__()
        self.emb = embedding_layer
        self.cnns = nn.ModuleList([torch.nn.Conv1d(in_channels=embedding_layer.embedding_dim,out_channels=n_filters,kernel_size=fs,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros') for fs in filter_sizes])
        self.linear = nn.Linear(len(filter_sizes)*n_filters, nclasses)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,batch_sentences):
        batch_embeddings, _ = self.emb(batch_sentences)
        x = batch_embeddings.transpose(1,2)
        x = [torch.relu(cnn(x).transpose(1,2)) for cnn in self.cnns]
        x = torch.cat([torch.max(conved,dim=1,keepdim=True)[0] for conved in x],dim=2).squeeze(dim=1)
        scores = self.linear(self.dropout(x))
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