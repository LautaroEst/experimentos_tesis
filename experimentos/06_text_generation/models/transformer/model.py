import torch
from torch import nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class TransformerDecoder(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self):
        pass


class TransformerEncoderDecoder(nn.Module):
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def init_transformer(d_model,nhead,dim_feedforward,dropout,num_layers):
    encoder = None
    decoder = None
    model = TransformerEncoderDecoder(encoder,decoder)

    return model



from transformers import AutoModel

if __name__ == "__main__":
    model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    print(model.encoder.layer[0])