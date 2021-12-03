import torch
from torch import nn


class LSTMEncoder(nn.Module):

    def __init__(self,tokenizer):
        super().__init__()
        vocab = tokenizer.vocab
        num_embeddings = len(vocab)
        pad_idx = vocab[tokenizer.pad_token]

        self.tokenizer = tokenizer
        self.emb = nn.Embedding(num_embeddings,4,padding_idx=pad_idx)

    def forward(self,src_input_ids,src_attention_mask):
        src_sent_enc = self.emb(src_input_ids)
        return src_sent_enc


class LSTMDecoder(nn.Module):

    def __init__(self,tokenizer):
        super().__init__()
        vocab = tokenizer.vocab
        num_embeddings = len(vocab)
        pad_idx = vocab[tokenizer.pad_token]

        self.tokenizer = tokenizer
        self.emb = nn.Embedding(num_embeddings,4,padding_idx=pad_idx)

    def forward(self,src_input_ids,src_attention_mask,tgt_input_ids,tgt_attention_mask):
        return src_input_ids.sum() + src_attention_mask.sum() + tgt_input_ids.sum() + tgt_attention_mask.sum()


class LSTMEncoderDecoder(nn.Module):

    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder = encoder.cuda(0)
        self.decoder = decoder.cuda(1)
    
    def forward(self,src_sent,tgt_sent):

        src_encoded_input = self.encoder.tokenizer(src_sent)
        tgt_encoded_input = self.decoder.tokenizer(tgt_sent)

        src_input_ids = src_encoded_input['input_ids'].cuda(0)
        src_attention_mask = src_encoded_input['attention_mask'].cuda(1)
        src_sent_enc = self.encoder(src_input_ids,src_attention_mask)

        src_sent_enc = src_sent_enc.cuda(1)
        src_attention_mask = src_encoded_input['attention_mask'].cuda(1)

        tgt_input_ids = tgt_encoded_input['input_ids'].cuda(1)
        tgt_attention_mask = tgt_encoded_input['attention_mask'].cuda(1)
        pred_sent = self.decoder(src_sent_enc,src_attention_mask,tgt_input_ids,tgt_attention_mask)
        return pred_sent


def init_lstm_model(src_tokenizer,tgt_tokenizer):
    encoder = LSTMEncoder(src_tokenizer)
    decoder = LSTMDecoder(tgt_tokenizer)
    model = LSTMEncoderDecoder(encoder,decoder)
    return model