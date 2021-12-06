import torch
from torch import nn
from torch.nn.modules.transformer import TransformerEncoder






class TransformerEncoderDecoder(nn.Module):
    
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        return self.encoder(x)


def init_transformer(d_model,nhead,dim_feedforward,dropout,num_layers):
    encoder_layer = nn.TransformerEncoderLayer(
                        d_model,nhead,dim_feedforward,
                        dropout,activation="relu",batch_first=True
    )
    encoder = nn.TransformerEncoder(encoder_layer,num_layers,norm=None)

    decoder_layer = nn.TransformerDecoderLayer(
                        d_model,nhead,dim_feedforward,
                        dropout,activation="relu",batch_first=True
    )
    decoder = nn.TransformerDecoder(decoder_layer,num_layers,norm=None)

    model = TransformerEncoderDecoder(encoder,decoder)

    return model



if __name__ == "__main__":

    print(torch.__version__)
    model = init_transformer(
        d_model=16,
        nhead=4,
        dim_feedforward=10,
        dropout=0.0,
        num_layers=4
    )
    x = torch.arange(128).view(2,4,16).float()
    model(x)