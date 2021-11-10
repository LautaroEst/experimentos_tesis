from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import numpy as np
import pandas as pd

class VocabVectorizer(object):

    def __init__(self,pattern,freq_cutoff,max_words,max_sent_len,
                pad_token,unk_token):

        self.pattern = pattern
        self.freq_cutoff = freq_cutoff
        self.max_words = max_words
        self.max_sent_len = max_sent_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = None

    def create_vocabulary(self,corpus):
        word_freq = defaultdict(lambda : 0)
        fc = self.freq_cutoff
        for sent in corpus:
            for word in sent:
                word_freq[word] += 1
        valid_words = [w for w, v in word_freq.items() if v >= fc]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:self.max_words-2]
        vocab = {word: idx for idx, word in enumerate(top_k_words,2)}
        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        self.vocab = vocab
        return vocab

    def fit_transform(self,ds):
        corpus = ds.str.findall(self.pattern)
        vocab = self.create_vocabulary(corpus)
        unk_idx = vocab[self.unk_token]
        max_sent_len = self.max_sent_len
        ds = corpus.apply(lambda sent: [vocab.get(tk,unk_idx) for tk in sent[:max_sent_len]])
        return ds
    
    def fit(self,ds):
        corpus = ds.str.findall(self.pattern)
        vocab = self.create_vocabulary(corpus)
        self.vocab = vocab

    def transform(self,ds):
        ds = ds.str.findall(self.pattern)
        vocab = self.vocab
        unk_idx = vocab[self.unk_token]
        ds = ds.apply(lambda sent: [vocab.get(tk,unk_idx) for tk in sent])
        return ds


class CNNModel(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,filter_sizes,n_filters,nclasses,dropout):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings,embedding_dim,padding_idx=0)
        self.cnns = nn.ModuleList([torch.nn.Conv1d(in_channels=embedding_dim,out_channels=n_filters,kernel_size=fs,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros') for fs in filter_sizes])
        self.linear = nn.Linear(len(filter_sizes)*n_filters, nclasses)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,x):
        x = self.emb(x).transpose(1,2)
        x = [torch.relu(cnn(x).transpose(1,2)) for cnn in self.cnns]
        x = torch.cat([torch.max(conved,dim=1,keepdim=True)[0] for conved in x],dim=2).squeeze(dim=1)
        scores = self.linear(self.dropout(x))
        return scores

def batch_iter(ds,y,batch_size,pad_idx):

    N = len(ds)
    #df = pd.concat((ds,pd.Series(y)),keys=['x','y'],axis=1)
    indices_batches = torch.randperm(N).split(batch_size)
    for indices in indices_batches:
        #batch = df.iloc[indices,:].sort_values(by=['x'],key=lambda x: x.str.len(),ascending=False)

        #sequence_batch, y_batch = batch['x'], batch['y'].values
        sequence_batch, y_batch = ds.iloc[indices].reset_index(drop=True), y[indices]
        #sent_lenghts = sequence_batch.str.len().tolist()
        #max_len = len(sequence_batch.iloc[0])
        max_len = sequence_batch.str.len().max()
        padded_sequences = [sent + [pad_idx] * (max_len-len(sent)) for sent in sequence_batch]
        padded_sequences = torch.LongTensor(padded_sequences)
        y_batch = torch.LongTensor(y_batch)
        #yield padded_sequences, sent_lenghts, y_batch
        yield padded_sequences, y_batch

                                

class Classifier(object):

    def __init__(self,nclasses,pattern,frequency_cutoff,max_tokens,max_sent_len,
                embedding_dim,filter_sizes,n_filters,dropout,batch_size,
                learning_rate,num_epochs,device):

        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.nclasses = nclasses
        self.device_type = device
        
        self.vec = VocabVectorizer(pattern,frequency_cutoff,
                        max_tokens,max_sent_len,'<pad>','<unk>')

    def check_accuracy(self, scores, y_batch):
        
        with torch.no_grad():
            y_pred = torch.argmax(scores,dim=1).cpu().numpy()
            y_true = y_batch.cpu().numpy()
        
        correct = (y_pred == y_true)
        acc = correct.mean()
        print('{}/{} ({:.2f}%)'.format(sum(correct),len(correct),acc * 100))
        return acc


    def train(self,ds,y,eval_every=1,dev=None):
        ds = self.normalize_dataset(ds)
        ds = self.vec.fit_transform(ds)
        pad_idx = self.vec.vocab[self.vec.pad_token]

        if dev:
            ds_dev = self.normalize_dataset(dev[0])
            ds_dev = self.vec.transform(ds_dev)
            dev = (ds_dev,dev[1])

        device = torch.device(self.device_type)    
        model = CNNModel(len(self.vec.vocab),self.embedding_dim,self.filter_sizes,
                        self.n_filters,self.nclasses,self.dropout)
        model.to(device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)

        train_loss_history = []
        train_accuracy_history = []
        if dev:
            dev_loss_history = []
            dev_accuracy_history = []
        
        for e in range(self.epochs):
            print('Epoch {}/{}'.format(e+1,self.epochs))
            for i, (sequence_batch, y_batch) in enumerate(batch_iter(ds,y,self.batch_size,pad_idx)):
                sequence_batch = sequence_batch.to(device=device)
                y_batch = y_batch.to(device=device)

                scores = model(sequence_batch)
                loss = criterion(scores,y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (e * self.epochs + i) % eval_every == 0:

                    print('Train loss: {:.5f}'.format(loss.item()))
                    train_loss_history.append(loss.item())
                    print('Train accuracy:',end=' ')
                    train_acc = self.check_accuracy(scores, y_batch)
                    train_accuracy_history.append(train_acc)

                    if dev:
                        ds_dev, y_dev = dev
                        N_dev = len(ds_dev)
                        idx = np.random.permutation(N_dev)[:self.batch_size]
                        ds_dev, y_dev = ds_dev.iloc[idx].reset_index(drop=True), y_dev[idx]
                        sequence_dev_batch, y_dev_batch = next(batch_iter(ds_dev,y_dev,self.batch_size,pad_idx))
                        sequence_dev_batch = sequence_dev_batch.to(device=device)
                        y_dev_batch = y_dev_batch.to(device=device)

                        model.eval()
                        with torch.no_grad():
                            scores = model(sequence_dev_batch)
                            loss = criterion(scores,y_dev_batch)
                            print('Dev loss: {:.5f}'.format(loss.item()))
                            dev_loss_history.append(loss.item())
                            print('Dev accuracy:',end=' ')
                            dev_acc = self.check_accuracy(scores, y_dev_batch)
                            dev_accuracy_history.append(dev_acc)

                        model.train()

                    print()

        model.eval()
        self.model = model

        if dev:
            history = {
                'train_loss': train_loss_history,
                'train_accuracy': train_accuracy_history,
                'dev_loss': dev_loss_history,
                'dev_accuracy': dev_accuracy_history,
                'eval_every': eval_every,
                'epochs': self.epochs,
                'batch_size': self.batch_size
            }
        else:
            history = {
                'train_loss': train_loss_history,
                'train_accuracy': train_accuracy_history,
                'eval_every': eval_every,
                'epochs': self.epochs,
                'batch_size': self.batch_size
            }
        return history
            


    def predict(self,ds):
        ds = self.normalize_dataset(ds)
        ds = self.vec.transform(ds)
        pad_idx = self.vec.vocab[self.vec.pad_token]
        device = torch.device(self.device_type)
        model = self.model
        model.eval()

        N = len(ds)
        indices_batches = torch.arange(N).split(self.batch_size)
        y_pred_batches = []
        for indices in indices_batches:
            sequence_batch = ds.iloc[indices].reset_index(drop=True)
            max_len = sequence_batch.str.len().max()
            padded_sequences = [sent + [pad_idx] * (max_len-len(sent)) for sent in sequence_batch]
            padded_sequences = torch.LongTensor(padded_sequences).to(device=device)

            with torch.no_grad():
                scores = model(padded_sequences)
                y_pred = torch.argmax(scores,dim=1).cpu().numpy()
                y_pred_batches.append(y_pred)
        
        y_pred = np.hstack(y_pred_batches)

        return y_pred
                

    def normalize_dataset(self,ds):
        # Pasamos a minúscula todo
        ds = ds.str.lower()
        # Sacamos todos los acentos
        for rep, rep_with in [('[óòÓöøôõ]','ó'), 
                              ('[áàÁäåâãÄ]','á'), 
                              ('[íìÍïîÏ]','í'), 
                              ('[éèÉëêÈ]','é'), 
                              ('[úÚùû]','ú'),
                              ('[ç¢Ç]','c'), 
                              ('[ý¥]','y'),
                              ('š','s'),
                              ('ß','b'),
                              ('\x08','')]:
            ds  = ds.str.replace(rep,rep_with,regex=True)
        return ds