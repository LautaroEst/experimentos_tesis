import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
import pandas as pd
from .utils import VocabVectorizer
import numpy as np


class RNNModel(nn.Module):

    def __init__(self,rnn,bidirectional,embedding_dim,num_embeddings,hidden_size,num_outs,num_layers,dropout):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings,embedding_dim,padding_idx=0)

        if 'RNN' in rnn:
            if rnn == 'RNNrelu':
                nonlinearity = 'relu'
            elif rnn == 'RNNtanh':
                nonlinearity = 'tanh'
            self.rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,
                    num_layers=num_layers,nonlinearity=nonlinearity,bias=True,
                    batch_first=True,dropout=dropout,bidirectional=bidirectional)

        elif rnn == 'LSTM':
            self.rnn = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,
                    num_layers=num_layers,bias=True,batch_first=True,
                    dropout=dropout,bidirectional=bidirectional)

        elif rnn == 'GRU':
            self.rnn = nn.GRU(input_size=embedding_dim,hidden_size=hidden_size,
                    num_layers=num_layers,bias=True,batch_first=True,
                    dropout=dropout,bidirectional=bidirectional)

        self.rnn_type = rnn
        self.in_linear = 2*hidden_size if bidirectional else hidden_size
        self.linear_out = nn.Linear(self.in_linear,num_outs)

        
    def forward(self,in_sequence,seq_len):
        emb_seq = self.emb(in_sequence)
        packed_seq = pack_padded_sequence(emb_seq,seq_len,batch_first=True)
        out, hidden = self.rnn(packed_seq)
        if self.rnn_type == 'LSTM':
            hidden = hidden[0]
        scores = self.linear_out(hidden.transpose(0,1)[:,-2:,:].reshape(-1,self.in_linear))
        #out = pad_packed_sequence(out,batch_first=True,padding_value=0)
        return scores
        

        
def batch_iter(ds,y,batch_size,pad_idx):

    N = len(ds)
    df = pd.concat((ds,pd.Series(y)),keys=['x','y'],axis=1)
    indices_batches = torch.randperm(N).split(batch_size)
    for indices in indices_batches:
        batch = df.iloc[indices,:].sort_values(by=['x'],key=lambda x: x.str.len(),ascending=False)

        sequence_batch, y_batch = batch['x'], batch['y'].values
        sent_lenghts = sequence_batch.str.len().tolist()
        max_len = len(sequence_batch.iloc[0])
        padded_sequences = [sent + [pad_idx] * (max_len-len(sent)) for sent in sequence_batch]
        padded_sequences = torch.LongTensor(padded_sequences)
        y_batch = torch.LongTensor(y_batch)
        yield padded_sequences, sent_lenghts, y_batch


class RNNClassifier(object):

    def __init__(self,nclasses,rnn,bidirectional,frequency_cutoff,max_tokens,max_sent_len,
                embedding_dim,hidden_size,num_layers,dropout,batch_size,
                learning_rate,num_epochs,device):

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.nclasses = nclasses
        self.device_type = device
        self.rnn = rnn
        self.bidirectional = bidirectional

        self.vec = VocabVectorizer(frequency_cutoff,
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
        model = RNNModel(self.rnn,self.bidirectional,self.embedding_dim,len(self.vec.vocab),
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
        
        num_batches = len(torch.arange(len(ds)).split(self.batch_size))
        for e in range(self.epochs):

            for i, (sequence_batch, seq_len, y_batch) in enumerate(batch_iter(ds,y,self.batch_size,pad_idx)):
                sequence_batch = sequence_batch.to(device=device)
                y_batch = y_batch.to(device=device)

                scores = model(sequence_batch,seq_len)
                loss = criterion(scores,y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (e * self.epochs + i) % eval_every == 0:

                    print('Batch {}/{}. Epoch {}/{}'.format(i,num_batches,e+1,self.epochs))
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
                        sequence_dev_batch, seq_dev_len, y_dev_batch = next(batch_iter(ds_dev,y_dev,self.batch_size,pad_idx))
                        sequence_dev_batch = sequence_dev_batch.to(device=device)
                        y_dev_batch = y_dev_batch.to(device=device)

                        model.eval()
                        with torch.no_grad():
                            scores = model(sequence_dev_batch,seq_dev_len)
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
            sent_lenghts = sequence_batch.str.len()
            sorted_idx = sent_lenghts.argsort()[::-1]
            resorted_idx = sorted_idx.argsort()
            sorted_sequence_batch = sequence_batch.iloc[sorted_idx].reset_index(drop=True)
            sorted_sent_lenghts = sent_lenghts.iloc[sorted_idx].tolist()
            max_len = sorted_sent_lenghts[0]
            padded_sequences = [sent + [pad_idx] * (max_len-len(sent)) for sent in sorted_sequence_batch]
            padded_sequences = torch.LongTensor(padded_sequences).to(device=device)

            with torch.no_grad():
                scores = model(padded_sequences,sorted_sent_lenghts)
                y_pred = torch.argmax(scores,dim=1).cpu().numpy()[resorted_idx]
                y_pred_batches.append(y_pred)
        
        y_pred = np.hstack(y_pred_batches)

        return y_pred