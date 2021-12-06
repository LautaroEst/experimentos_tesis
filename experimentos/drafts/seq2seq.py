import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplicativeCrossAttention(nn.Module):

    def __init__(self,query_size,keyvalue_size):
        super().__init__()
        self.weight = nn.Parameter((torch.rand(query_size,2*keyvalue_size)*2-1) / torch.sqrt(query_size) )

    def forward(self,keyvalues,keyvalues_mask,query):
        torch.bmm(keyvalues)
        attn_scores = F.softmax(logits,dim=2)
        attn_hiddens = torch.matmul(attn_scores,encoder_outs)
        hidden_with_attn = torch.cat((attn_hiddens,decoder_hiddens),dim=2)
        return hidden_with_attn  

def main():
    # o = [torch.arange(32).view(4,8), torch.arange(32).view(4,8)*2, torch.arange(32).view(4,8)*3]
    # # seq = torch.cat([o_t.unsqueeze(1) for o_t in o],dim=1)#.view(2,3,4)
    # # print(seq)
    # seq = torch.cat(o,dim=1).view(4,3,8)
    # print(seq)
    # tgt_seq = torch.tensor([[0,2,2],[3,1,1],[1,1,4],[0,1,0]]).long().unsqueeze(-1)
    # p = torch.gather(seq,index=tgt_seq,dim=-1)
    # print(seq.size(), tgt_seq.size())
    # print(p)

    # t = torch.randint(0,4,(1,4,5))
    # print(t.max(dim=2).indices)
    t = torch.tensor([[1.]])
    print(t.item())

    # h_n = torch.arange(24).view(2,3,4)
    # print(h_n)
    # # print(h_n.permute(1,0,2).reshape(-1,2*4))
    # print(h_n.transpose(0,1).reshape(-1,2*4))

    # attn = MultiplicativeCrossAttention(2,4)
    # W = torch.ones(2,4).float()
    # query = torch.arange(6).view(3,2).float()
    # keyvalues = torch.arange(36).view(3,3,4).float()
    # mask = torch.tensor([[1,1,1],[1,1,0],[1,1,0]]).bool()
    # print(keyvalues)
    # print(W)
    # e_t = torch.bmm(torch.matmul(keyvalues,W.T),query.unsqueeze(2))
    # e_t.data.masked_fill_(~mask.unsqueeze(2), -float('inf'))
    # alpha_t = F.softmax(e_t,dim=1)
    # a_t = torch.bmm(keyvalues.transpose(2,1),alpha_t).squeeze(2)
    # print(a_t)







if __name__ == "__main__":
    main()