from collections import defaultdict
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from transformers import BertConfig, BertTokenizer, BertForSequenceClassification

        
def batch_iter(input_ids,token_type_ids,attention_mask,labels,batch_size,pad_idx,shuffle=True):

    N = len(input_ids)
    if shuffle:
        indices_batches = torch.randperm(N).split(batch_size)
    else:
        indices_batches = torch.arange(N).split(batch_size)

    for indices in tqdm(indices_batches):
        input_ids_batch = input_ids.iloc[indices].reset_index(drop=True)
        token_type_ids_batch = token_type_ids.iloc[indices].reset_index(drop=True)
        attention_mask_batch = attention_mask.iloc[indices].reset_index(drop=True)
        labels_batch = labels[indices]

        max_len = input_ids_batch.str.len().max()

        for input_sent, token_sent, attention_sent in zip(input_ids_batch,token_type_ids_batch,attention_mask_batch):
            sent_len = len(input_sent)
            input_sent.extend([pad_idx] * (max_len - sent_len))
            token_sent.extend([0] * (max_len - sent_len))
            attention_sent.extend([0] * (max_len - sent_len))
        input_ids_batch_padded = torch.LongTensor(input_ids_batch)
        token_type_ids_batch_padded = torch.LongTensor(token_type_ids_batch)
        attention_mask_batch_padded = torch.LongTensor(attention_mask_batch)
        labels_batch = torch.from_numpy(labels_batch).long()
        yield (input_ids_batch_padded, token_type_ids_batch_padded, 
            attention_mask_batch_padded, labels_batch)


class Classifier(object):

    def __init__(self,nclasses,model_path,cased,dropout,batch_size,learning_rate,num_epochs,device):

        self.model_path = model_path
        self.dropout = dropout
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.nclasses = nclasses
        self.device_type = device
        self.cased = cased

        with open(model_path + 'config.json', 'r') as f:
            config_json = json.load(f)
        config = BertConfig(
            vocab_size=config_json['vocab_size'],
            hidden_size=config_json['hidden_size'],
            num_hidden_layers=config_json['num_hidden_layers'],
            num_attention_heads=config_json['num_attention_heads'],
            intermediate_size=config_json['intermediate_size'],
            hidden_act=config_json['hidden_act'],
            hidden_dropout_prob=config_json['hidden_dropout_prob'],
            attention_probs_dropout_prob=config_json['attention_probs_dropout_prob'],
            max_position_embeddings=config_json['max_position_embeddings'],
            type_vocab_size=config_json['type_vocab_size'],
            initializer_range=config_json['initializer_range'],
            position_embedding_type='absolute',
            # Parámetros de clasificación:
            classifier_dropout=dropout,
            num_labels=nclasses
        )

        self.config = config
        self.max_len = config_json['max_position_embeddings']
        do_lower_case = False if cased else True
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case=do_lower_case)


    def check_accuracy(self, scores, y_batch):
        
        with torch.no_grad():
            y_pred = torch.argmax(scores,dim=1).cpu().numpy()
            y_true = y_batch.cpu().numpy()
        
        correct = (y_pred == y_true)
        acc = correct.mean()
        print('{}/{} ({:.2f}%)'.format(sum(correct),len(correct),acc * 100))
        return acc


    def train(self,ds,y,eval_every=1,dev=None):
        ds = self.normalize_dataset(ds,self.cased).tolist()
        encoded_input = self.tokenizer(ds,truncation=True,max_length=self.max_len)
        input_ids = pd.Series(encoded_input['input_ids'])
        token_type_ids = pd.Series(encoded_input['token_type_ids'])
        attention_mask = pd.Series(encoded_input['attention_mask'])
        pad_idx = self.tokenizer.pad_token_id

        if dev:
            ds_dev = self.normalize_dataset(dev[0],self.cased).tolist()
            encoded_dev_input = self.tokenizer(ds_dev,truncation=True,max_length=self.max_len)
            dev = (
                pd.Series(encoded_dev_input['input_ids']),
                pd.Series(encoded_dev_input['token_type_ids']),
                pd.Series(encoded_dev_input['attention_mask']),
                dev[1]
            )

        device = torch.device(self.device_type)    
        model = BertForSequenceClassification.from_pretrained(self.model_path,config=self.config)
        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)

        train_loss_history = []
        train_accuracy_history = []
        if dev:
            dev_loss_history = []
            dev_accuracy_history = []
        
        num_batches = len(torch.arange(len(input_ids)).split(self.batch_size))
        for e in range(self.epochs):
            print('Epoch {}/{}'.format(e+1,self.epochs))
            for i, batch in enumerate(batch_iter(
                                            input_ids,
                                            token_type_ids,
                                            attention_mask,
                                            y,
                                            self.batch_size,
                                            pad_idx
                                        )):
                
                input_ids_batch, token_type_ids_batch, attention_mask_batch, labels_batch = (x.to(device=device) for x in batch)
                
                scores = model(input_ids=input_ids_batch,
                            attention_mask=attention_mask_batch,
                            token_type_ids=token_type_ids_batch,
                            labels=labels_batch,
                            output_hidden_states=False,
                            output_attentions=False,
                            return_dict=True)
                loss = scores['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if (e * self.epochs + i) % eval_every == 0:

                    print('Batch {}/{}. Epoch {}/{}'.format(i,num_batches,e+1,self.epochs))
                    print('Train loss: {:.5f}'.format(loss.item()))
                    train_loss_history.append(loss.item())
                    print('Train accuracy:',end=' ')
                    train_acc = self.check_accuracy(scores['logits'], labels_batch)
                    train_accuracy_history.append(train_acc)

                    if dev:
                        input_ids_dev, token_type_ids_dev, attention_mask_dev, y_dev = dev
                        N_dev = len(input_ids_dev)
                        idx = np.random.permutation(N_dev)[:self.batch_size]
                        input_ids_dev = input_ids_dev.iloc[idx].reset_index(drop=True)
                        token_type_ids_dev = token_type_ids_dev.iloc[idx].reset_index(drop=True)
                        attention_mask_dev = attention_mask_dev.iloc[idx].reset_index(drop=True)
                        y_dev =  y_dev[idx]
                        batch = next(batch_iter(
                                            input_ids_dev,
                                            token_type_ids_dev,
                                            attention_mask_dev,
                                            y_dev,
                                            self.batch_size,
                                            pad_idx
                                        ))
                        (input_ids_dev, token_type_ids_dev, attention_mask_dev, y_dev) =  (x.to(device=device) for x in batch)

                        model.eval()
                        with torch.no_grad():
                            scores = model(
                                input_ids=input_ids_dev,
                                attention_mask=token_type_ids_dev,
                                token_type_ids=attention_mask_dev,
                                labels=y_dev,
                                output_hidden_states=False,
                                output_attentions=False,
                                return_dict=True
                            )
                            loss = scores['loss']
                            print('Dev loss: {:.5f}'.format(loss.item()))
                            dev_loss_history.append(loss.item())
                            print('Dev accuracy:',end=' ')
                            dev_acc = self.check_accuracy(scores['logits'], y_dev)
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
        ds = self.normalize_dataset(ds,self.cased).tolist()
        encoded_input = self.tokenizer(ds,truncation=True,max_length=self.max_len)
        input_ids = pd.Series(encoded_input['input_ids'])
        token_type_ids = pd.Series(encoded_input['token_type_ids'])
        attention_mask = pd.Series(encoded_input['attention_mask'])
        pad_idx = self.tokenizer.pad_token_id
        labels = np.zeros(len(input_ids))

        device = torch.device(self.device_type)
        model = self.model
        model.eval()

        num_batches = len(torch.arange(len(input_ids)).split(self.batch_size))
        y_pred_batches = []
        for batch in tqdm(batch_iter(input_ids,token_type_ids,attention_mask,labels,self.batch_size,pad_idx,shuffle=False),total=num_batches):
            input_ids_batch, token_type_ids_batch, attention_mask_batch, y_batch = (x.to(device=device) for x in batch)
            
            with torch.no_grad():
                scores = model(
                    input_ids=input_ids_batch,
                    attention_mask=token_type_ids_batch,
                    token_type_ids=attention_mask_batch,
                    labels=y_batch,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True
                )
                y_pred = torch.argmax(scores['logits'],dim=1).cpu().numpy()
                y_pred_batches.append(y_pred)
        
        y_pred = np.hstack(y_pred_batches)

        return y_pred
                

    def normalize_dataset(self,ds,cased):
        if not cased:
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