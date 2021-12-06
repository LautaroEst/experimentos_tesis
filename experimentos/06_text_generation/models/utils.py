import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm



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


def greedy_decoding(model,src_sents,tgt_tokenizer,max_len):

    was_training = model.training    
    model.eval()

    pred_sents = []
    tokens_to_be_removed = {
        tgt_tokenizer.pad_token, 
        tgt_tokenizer.start_token, 
        tgt_tokenizer.end_token
    }
    with torch.no_grad():
        for src_sent in tqdm(src_sents):
            pred_ids = model.greedy_decode(src_sent,max_len)
            pred_tokens = tgt_tokenizer.ids_to_tokens(pred_ids)
            pred_sent = [tk for tk in pred_tokens if tk not in tokens_to_be_removed]
            pred_sents.append(pred_sent)

    if was_training:
        model.train()

    return pred_sents