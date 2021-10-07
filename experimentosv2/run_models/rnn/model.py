from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


def create_vocabulary(corpus,freq_cutoff,max_words,pad_token,unk_token):
    word_freq = defaultdict(lambda : 0)
    for sent in corpus:
        for word in sent:
            word_freq[word] += 1
    valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
    top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:max_words-1]
    vocab = {word: idx for idx, word in enumerate(top_k_words,2)}
    vocab[pad_token] = 0
    vocab[unk_token] = 1
    return vocab


class RNNModel(nn.Module):

    def __init__(self,embedding_dim,num_embeddings,hidden_size,num_layers,dropout):
        self.emb = nn.Embedding(num_embeddings,embedding_dim,padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,
                   num_layers=num_layers,nonlinearity='relu',bias=True,
                   batch_first=True,dropout=dropout,bidirectional=False)
        
    def forward(self,in_sequence):
        pass

        
def batch_iter(ds,y,vocab,batch_size):

    N = len(ds)
    num_batchs = (N // batch_size) + 1 if N % batch_size == 0 else N // batch_size
    indices_batches = torch.randperm(N).split(batch_size)
    for indices in indices_batches:
        sequence_batch = ds.iloc[indices].sort_values(key=lambda x: x.str.len(),ascending=False)
        y_batch = 
        max_len = len(batch[0])
        padded_sequences = [[vocab[tk] for tk in sent] + \
                            [0] * (max_len-len(sent)) for sent in batch]
        padded_sequences = torch.tensor(padded_sequences,dtype=torch.float)
        yield padded_sequences

                                
    




class Classifier(object):

    def __init__(self):
        self.freq_cutoff = 5
        self.max_words = 10000
        self.max_len = 128
        self.embedding_dim = 300
        self.hidden_size = 200
        self.num_layers = 1
        self.dropout = 0.
        self.batch_size = 128

    def train(self,ds,y):
        ds = ds.str.findall(r'(\w+|[\.,!\(\)"\-:\?/%;¡\$\'¿\\]|\d+)')
        vocab = create_vocabulary(ds,self.freq_cutoff,self.max_words)
        
        model = RNNModel(self.embedding_dim,len(self.vocab),
                self.hidden_size,self.num_layers,self.dropout)

        for batch in batch_iter(ds,vocab,self.batch_size):
            pass

    def predict(self,ds):
        pass

    def normalize_dataset(self,ds):
        # Pasamos a minúscula todo
        ds = ds.str.lower()
        # Sacamos todos los acentos
        for rep, rep_with in [('[óòÓöøôõ]','o'), ('[áàÁäåâãÄ]','a'), ('[íìÍïîÏ]','i'), 
                            ('[éèÉëêÈ]','e'), ('[úüÚùûÜ]','u'), ('[ç¢Ç]','c'), 
                            ('[ý¥]','y'),('š','s'),('ß','b'),('\x08','')]:
            ds  = ds.str.replace(rep,rep_with,regex=True)
        return ds