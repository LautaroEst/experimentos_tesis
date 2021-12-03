import torch
from torch import nn


class LSTMEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10,4,padding_idx=0)

    def forward(self,src_sent):
        src_sent_ids = src_sent
        src_sent_enc = self.emb(src_sent_ids)
        return src_sent_enc


class LSTMDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,2)

    def forward(self,src_sent_enc,tgt_sent):
        tgt_sent_enc = tgt_sent
        pred_sent = self.linear(src_sent_enc)
        return pred_sent


class LSTMEncoderDecoder(nn.Module):

    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder.cuda(0)
        self.decoder = decoder.cuda(1)
    
    def forward(self,src_sent,tgt_sent):
        src_sent = src_sent.cuda(0)
        src_sent_enc = self.encoder(src_sent)
        src_sent_enc, tgt_sent = src_sent_enc.cuda(1), tgt_sent.cuda(1)
        pred_sent = self.decoder(src_sent_enc, tgt_sent)
        return pred_sent


def init_lstm_model():
    encoder = LSTMEncoder()
    decoder = LSTMDecoder()
    model = LSTMEncoderDecoder(encoder,decoder)
    return model