from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import numpy as np
import pandas as pd


class CatBOWVectorizer(object):

    def __init__(self,nclasses,pattern,ngram_range,min_df,max_features):
        self.vec = TfidfVectorizer(token_pattern=pattern,
                ngram_range=ngram_range,min_df=min_df,
                max_features=max_features)
        self.labels_names = list(range(nclasses))
    
    def _vectorizer_by_cat(self,X,labels):
        n_cats = len(self.labels_names)
        word_vecs = np.zeros((X.shape[1],n_cats),dtype=float)
        for i,n in enumerate(self.labels_names):
                word_vecs[:,i] = X[labels == n,:].sum(axis=0)
        X_cats = X @ word_vecs
        self.learned_words = word_vecs
        return X_cats

    def fit_transform(self,corpus,labels):
        X = self.vec.fit_transform(corpus)
        X_cats = self._vectorizer_by_cat(X,labels)
        self.learned_mean = X_cats.mean(axis=0,keepdims=True)
        X_cats = X_cats - self.learned_mean
        return X_cats
    
    def transform(self,corpus):
        X = self.vec.transform(corpus)
        X_cats = X @ self.learned_words
        X_cats = X_cats - self.learned_mean
        return X_cats


class TwoLayerNet(nn.Module):
    def __init__(self,input_features,hidden_features,output_features):
        super().__init__()
        self.linear1 = nn.Linear(input_features,hidden_features,bias=True)
        self.linear2 = nn.Linear(hidden_features,output_features,bias=True)
    
    def forward(self,x):
        x = torch.relu(self.linear1(x))
        log_probs = self.linear2(x)
        return log_probs


class Classifier(object):

    def __init__(self,nclasses,pattern,ngram_range,min_df,max_features,
                hidden_size,num_epochs,batch_size,learning_rate,weight_decay,
                device='cuda:1'):

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = num_epochs
        self.nclasses = nclasses
        self.device_type = device

        self.vec = CatBOWVectorizer(nclasses,pattern,ngram_range,min_df,max_features)

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
        X_train = self.vec.fit_transform(ds,y)
        X_train = torch.from_numpy(X_train).type(torch.float)
        y_train = torch.LongTensor(y)

        train_dataset = TensorDataset(X_train,y_train)
        train_dataloader = DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)

        if dev:
            ds_dev = self.normalize_dataset(dev[0])
            X_dev = self.vec.transform(ds_dev)
            X_dev, y_dev = torch.from_numpy(X_dev).type(torch.float), torch.LongTensor(dev[1])
            dev = (X_dev,y_dev)

        device = torch.device(self.device_type)    
        model = TwoLayerNet(self.nclasses,self.hidden_size,self.nclasses)
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
            for i, batch in enumerate(train_dataloader):
                X_batch, y_batch = (x.to(device=device) for x in batch)

                scores = model(X_batch)
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
                        X_dev, y_dev = dev
                        N_dev = X_dev.size(0)
                        idx = np.random.permutation(N_dev)[:self.batch_size]
                        X_dev_batch, y_dev_batch = (x.to(device=device) for x in (X_dev[idx,:], y_dev[idx]))

                        model.eval()
                        with torch.no_grad():
                            scores = model(X_dev_batch)
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
        X = self.vec.transform(ds)
        X = torch.from_numpy(X).type(torch.float)
        device = torch.device(self.device_type)
        model = self.model
        model.eval()

        N = X.size(0)
        indices_batches = torch.arange(N).split(self.batch_size)
        y_pred_batches = []
        for indices in indices_batches:
            X_batch = X[indices,:].to(device=device)

            with torch.no_grad():
                scores = model(X_batch)
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