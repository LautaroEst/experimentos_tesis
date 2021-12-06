from collections import defaultdict
import re

import torch
from torch._C import Value


class WordTokenizer(object):

    pattern = r"(\w+|[\.,!\(\)\"\-:\?/%;¡\$'¿\\]|\d+)"

    def __init__(
            self,freq_cutoff,max_tokens,max_sent_len,
            pad_token,unk_token,start_token,end_token
        ):

        self.freq_cutoff = freq_cutoff
        self.max_tokens = max_tokens
        self.max_sent_len = max_sent_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token = start_token
        self.end_token = end_token
        self.vocab = None
        self.inv_vocab = None

    def create_vocabulary(self,corpus):
        word_freq = defaultdict(lambda : 0)
        fc = self.freq_cutoff
        for sent in corpus:
            for word in sent:
                word_freq[word] += 1
        valid_words = [w for w, v in word_freq.items() if v >= fc]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:self.max_tokens-2]
        vocab = {word: idx for idx, word in enumerate(top_k_words,4)}
        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        vocab[self.start_token] = 2
        vocab[self.end_token] = 3
        self.vocab = vocab
        self.inv_vocab = {idx:tk for tk, idx in vocab.items()}
        return vocab

    def ids_to_tokens(self,ids):
        return [self.inv_vocab[idx] for idx in ids]

    def tokenize(self,string):
        vocab = self.vocab
        unk_token = self.unk_token
        pattern = re.compile(self.pattern)
        return [tk if tk in vocab else unk_token for tk in pattern.findall(string)]

    def pre_tokenize(self,ds):
        return ds.str.findall(self.pattern)

    @classmethod
    def from_dataseries(cls,ds,**kwargs):
        tokenizer = cls(**kwargs)
        ds = tokenizer.pre_tokenize(ds)
        _ = tokenizer.create_vocabulary(ds)
        return tokenizer

    def __call__(self,ds):

        vocab = self.vocab
        unk_idx = vocab[self.unk_token]
        pad_idx = vocab[self.pad_token]
        start_idx = vocab[self.start_token]
        end_idx = vocab[self.end_token]
        max_sent_len = self.max_sent_len

        ds = self.pre_tokenize(ds)

        inputs_ids = []
        max_len = min([ds.str.len().max()+2,max_sent_len])
        for sent in ds:
            input_ids = [vocab.get(tk,unk_idx) for tk in sent[:(max_sent_len-2)]]
            input_ids.insert(0,start_idx)
            input_ids.append(end_idx)
            input_ids.extend([pad_idx] * (max_len-len(input_ids)))
            inputs_ids.append(input_ids)

        input_ids = torch.LongTensor(inputs_ids)
        attention_mask = (input_ids != pad_idx).float()
        encoded_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        return encoded_input