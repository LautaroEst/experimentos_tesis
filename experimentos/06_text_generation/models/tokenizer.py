from collections import defaultdict

import torch


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

    def create_vocabulary(self,corpus):
        word_freq = defaultdict(lambda : 0)
        fc = self.freq_cutoff
        for sent in corpus:
            for word in sent:
                word_freq[word] += 1
        valid_words = [w for w, v in word_freq.items() if v >= fc]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:self.max_tokens-2]
        vocab = {word: idx for idx, word in enumerate(top_k_words,2)}
        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        self.vocab = vocab
        return vocab

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
        max_sent_len = self.max_sent_len

        ds = self.pre_tokenize(ds)

        encoded_input = {
            "input_ids": [],
            "attention_mask": []
        }

        max_len = ds.str.len().max()
        for sent in ds:
            input_ids = []
            attention_mask = []
            for tk in sent[:max_sent_len]:
                input_ids.append(vocab.get(tk,unk_idx))
                attention_mask.append(1)
            input_ids.extend([pad_idx] * (max_len-len(sent)))
            attention_mask.extend([0] * (max_len-len(sent)))
            encoded_input['input_ids'].append(input_ids)
            encoded_input['attention_mask'].append(attention_mask)

        encoded_input['input_ids'] = torch.LongTensor(encoded_input['input_ids'])
        encoded_input['attention_mask'] = torch.LongTensor(encoded_input['attention_mask'])

        return encoded_input