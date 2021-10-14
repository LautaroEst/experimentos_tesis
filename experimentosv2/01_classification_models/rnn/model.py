from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import numpy as np
import pandas as pd


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

    def __init__(self,embedding_dim,num_embeddings,hidden_size,num_outs,num_layers,dropout):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings,embedding_dim,padding_idx=0)
        self.rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,
                   num_layers=num_layers,nonlinearity='relu',bias=True,
                   batch_first=True,dropout=dropout,bidirectional=False)
        self.linear_out = nn.Linear(hidden_size,num_outs)
        
    def forward(self,in_sequence,seq_len):
        emb_seq = self.emb(in_sequence)
        packed_seq = pack_padded_sequence(emb_seq,seq_len,batch_first=True)
        out, hidden = self.rnn(packed_seq)
        scores = self.linear_out(hidden.transpose(0,1)[:,-1,:])
        #out = pad_packed_sequence(out,batch_first=True,padding_value=0)
        return scores
        

        
def batch_iter(ds,y,vocab,batch_size):

    N = len(ds)
    df = pd.concat((ds,pd.Series(y)),keys=['x','y'],axis=1)
    indices_batches = torch.randperm(N).split(batch_size)
    for indices in indices_batches:
        batch = df.iloc[indices,:].sort_values(by=['x'],key=lambda x: x.str.len(),ascending=False)

        sequence_batch, y_batch = batch['x'], batch['y'].values
        sent_lenghts = sequence_batch.str.len().tolist()
        max_len = len(sequence_batch.iloc[0])
        padded_sequences = [[vocab.get(tk,1) for tk in sent] + \
                            [0] * (max_len-len(sent)) for sent in sequence_batch]
        padded_sequences = torch.LongTensor(padded_sequences)
        y_batch = torch.LongTensor(y_batch)
        yield padded_sequences, sent_lenghts, y_batch

                                
    




class Classifier(object):

    def __init__(self,nclasses):
        self.freq_cutoff = 5
        self.max_words = 10000
        self.max_len = 128
        self.embedding_dim = 300
        self.hidden_size = 200
        self.num_layers = 1
        self.dropout = 0.
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.epochs = 1
        self.nclasses = nclasses
        self.device_type = 'cuda:1'

    def check_accuracy(self,model,sequence_batch, seq_len, y_batch):
        model.eval()

        with torch.no_grad():
            scores = model(sequence_batch,seq_len)
            y_pred = torch.argmax(scores,dim=1).cpu().numpy()
            y_true = y_batch.cpu().numpy()
        
        correct = (y_pred == y_true)
        acc = correct.mean()
        print('{}/{} ({:.2f}%)'.format(sum(correct),len(correct),acc * 100))
        model.train()
        return acc


    def train(self,ds,y,eval_every=1,dev=None):
        pattern = r'(\w+|[\.,!\(\)"\-:\?/%;¡\$\'¿\\]|\d+)'
        ds = self.normalize_dataset(ds)
        ds = ds.str.findall(pattern)
        self.vocab = create_vocabulary(ds,self.freq_cutoff,self.max_words,'<pad>','<unk>')

        if dev:
            ds_dev = self.normalize_dataset(dev[0]).str.findall(pattern)
            dev = (ds_dev,dev[1])

        device = torch.device(self.device_type)    
        model = RNNModel(self.embedding_dim,len(self.vocab),
                self.hidden_size,self.nclasses,self.num_layers,self.dropout)
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
            for i, (sequence_batch, seq_len, y_batch) in enumerate(batch_iter(ds,y,self.vocab,self.batch_size)):
                sequence_batch = sequence_batch.to(device=device)
                y_batch = y_batch.to(device=device)

                scores = model(sequence_batch,seq_len)
                loss = criterion(scores,y_batch)
                
                if (e * self.epochs + i) % eval_every == 0:
                    print('Train loss: {:.5f}'.format(loss.item()))
                    train_loss_history.append(loss.item())
                    print('Train accuracy:',end=' ')
                    train_acc = self.check_accuracy(model,sequence_batch, seq_len, y_batch)
                    train_accuracy_history.append(train_acc)
                    if dev:
                        ds_dev, y_dev = dev
                        N_dev = len(ds_dev)
                        idx = np.random.permutation(N_dev)[:self.batch_size]
                        ds_dev, y_dev = ds_dev.iloc[idx].reset_index(drop=True), y_dev[idx]
                        sequence_dev_batch, seq_dev_len, y_dev_batch = next(batch_iter(ds_dev,y_dev,self.vocab,self.batch_size))
                        sequence_dev_batch = sequence_dev_batch.to(device=device)
                        y_dev_batch = y_dev_batch.to(device=device)
                        scores = model(sequence_dev_batch,seq_dev_len)
                        loss = criterion(scores,y_dev_batch)
                        print('Dev loss: {:.5f}'.format(loss.item()))
                        dev_loss_history.append(loss.item())
                        print('Dev accuracy:',end=' ')
                        dev_acc = self.check_accuracy(model,sequence_dev_batch, seq_dev_len, y_dev_batch)
                        dev_accuracy_history.append(dev_acc)
                    print()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        model.eval()
        self.model = model

        if dev:
            history = {
                'train_loss': train_loss_history,
                'train_accuracy': train_accuracy_history,
                'dev_loss': dev_loss_history,
                'dev_accuracy': dev_accuracy_history,
            }
        else:
            history = {
                'train_loss': train_loss_history,
                'train_accuracy': train_accuracy_history
            }
        return history
            


    def predict(self,ds):
        pattern = r'(\w+|[\.,!\(\)"\-:\?/%;¡\$\'¿\\]|\d+)'
        ds = self.normalize_dataset(ds)
        ds = ds.str.findall(pattern)
        device = torch.device(self.device_type)
        vocab = self.vocab
        model = self.model
        model.eval()

        N = len(ds)
        indices_batches = torch.arange(N).split(self.batch_size)
        y_pred_batches = []
        for indices in indices_batches:
            sequence_batch = ds.iloc[indices].sort_values(key=lambda x: x.str.len(),ascending=False)
            sent_lenghts = sequence_batch.str.len().tolist()
            max_len = len(sequence_batch.iloc[0])
            padded_sequences = [[vocab.get(tk,1) for tk in sent] + \
                                [0] * (max_len-len(sent)) for sent in sequence_batch]
            padded_sequences = torch.LongTensor(padded_sequences).to(device=device)

            with torch.no_grad():
                scores = model(padded_sequences,sent_lenghts)
                y_pred = torch.argmax(scores,dim=1).cpu().numpy()
                y_pred_batches.append(y_pred)
        
        y_pred = np.hstack(y_pred_batches)
        return y_pred
                

            





    def normalize_dataset(self,ds):
        # Pasamos a minúscula todo
        ds = ds.str.lower()
        # Sacamos todos los acentos
        for rep, rep_with in [('[óòÓöøôõ]','o'), ('[áàÁäåâãÄ]','a'), ('[íìÍïîÏ]','i'), 
                            ('[éèÉëêÈ]','e'), ('[úüÚùûÜ]','u'), ('[ç¢Ç]','c'), 
                            ('[ý¥]','y'),('š','s'),('ß','b'),('\x08','')]:
            ds  = ds.str.replace(rep,rep_with,regex=True)
        return ds