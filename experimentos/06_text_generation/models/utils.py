import torch
from torch import nn
from torch.nn import functional as F

class MultiplicativeCrossAttention(nn.Module):

    def __init__(self,query_size,keyvalue_size):
        super().__init__()
        self.weight = nn.Parameter((torch.rand(query_size,keyvalue_size)*2-1) / query_size )

    def forward(self,query,keyvalues,keyvalues_mask):
        e_t = torch.bmm(torch.matmul(keyvalues,self.weight.T),query.unsqueeze(2))
        keyvalues_mask = keyvalues_mask.bool()
        e_t.data.masked_fill_(~keyvalues_mask.unsqueeze(2), -float('inf'))
        alpha_t = F.softmax(e_t,dim=1)
        a_t = torch.bmm(keyvalues.transpose(2,1),alpha_t).squeeze(2)
        return a_t    