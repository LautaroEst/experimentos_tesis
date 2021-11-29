import os
import json
import numpy as np

import torch
from torch import nn, optim
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm


supported_models = [ 
    "dccuchile/bert-base-spanish-wwm-uncased",
    "Hate-speech-CNERG/dehatebert-mono-spanish",
]


def batch_iter(encoded_inputs,batch_size,pad_idx,labels=None,shuffle=True):

    N = len(encoded_inputs)
    if shuffle:
        indices_batches = torch.randperm(N).split(batch_size)
    else:
        indices_batches = torch.arange(N).split(batch_size)

    for indices in tqdm(indices_batches):
        input_ids_batch = encoded_inputs.loc[indices,'input_ids'].reset_index(drop=True)
        token_type_ids_batch = encoded_inputs.loc[indices,'token_type_ids'].reset_index(drop=True)
        attention_mask_batch = encoded_inputs.loc[indices,'attention_mask'].reset_index(drop=True)

        max_len = input_ids_batch.str.len().max()

        for input_sent, token_sent, attention_sent in zip(input_ids_batch,token_type_ids_batch,attention_mask_batch):
            sent_len = len(input_sent)
            input_sent.extend([pad_idx] * (max_len - sent_len))
            token_sent.extend([0] * (max_len - sent_len))
            attention_sent.extend([0] * (max_len - sent_len))

        input_ids_batch_padded = torch.LongTensor(input_ids_batch)
        token_type_ids_batch_padded = torch.LongTensor(token_type_ids_batch)
        attention_mask_batch_padded = torch.LongTensor(attention_mask_batch)
        
        # print(input_ids_batch_padded.size())
        # print(token_type_ids_batch_padded.size())
        # print(attention_mask_batch_padded.size())

        labels_batch = torch.from_numpy(np.asarray(labels[indices])).long()
        
        yield (input_ids_batch_padded, token_type_ids_batch_padded, 
            attention_mask_batch_padded, labels_batch)



class TransformerClassifier(object):

    def __init__(self,model_src,nclasses,dropout,batch_size,learning_rate,num_epochs,device):

        self.model_src = model_src
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.nclasses = nclasses
        self.device_type = device
        
        self.config = AutoConfig.from_pretrained(
            model_src,
            _num_labels=nclasses,
            classifier_dropout=dropout,
            id2label={i:str(i) for i in range(nclasses)},
            label2id={str(i):i for i in range(nclasses)}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_src,
            config=self.config
        )
        self.config.pad_token_id = self.tokenizer.pad_token_id
        self.max_len = self.config.max_position_embeddings


    def check_accuracy(self, scores, y_batch):
        
        with torch.no_grad():
            y_pred = torch.argmax(scores,dim=1).cpu().numpy()
            y_true = y_batch.cpu().numpy()
        
        correct = (y_pred == y_true)
        acc = correct.mean()
        print('{}/{} ({:.2f}%)'.format(sum(correct),len(correct),acc * 100))
        return acc


    def train(self,ds,y,eval_every=1,dev=None):

        ds = ds.tolist()
        encoded_input = self.tokenizer(
            ds,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        encoded_inputs = pd.DataFrame.from_dict({
            'input_ids': encoded_input.input_ids,
            'token_type_ids': encoded_input.token_type_ids,
            'attention_mask': encoded_input.attention_mask
        })
        pad_idx = self.tokenizer.pad_token_id

        if dev:
            ds_dev = dev[0].tolist()
            encoded_dev_input = self.tokenizer(
                ds_dev,
                truncation=True,
                max_length=self.max_len,
                return_token_type_ids=True,
                return_attention_mask=True
            )
            encoded_dev_inputs = pd.DataFrame.from_dict({
                'input_ids': encoded_dev_input.input_ids,
                'token_type_ids': encoded_dev_input.token_type_ids,
                'attention_mask': encoded_dev_input.attention_mask
            })
            dev = (encoded_dev_inputs, dev[1])

        device = torch.device(self.device_type)    
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_src,
            config=self.config,
            ignore_mismatched_sizes=True
        )
        model.to(device)
        model.train()

        optimizer = optim.Adam(model.parameters(),lr=self.learning_rate)

        train_loss_history = []
        train_accuracy_history = []
        if dev:
            dev_loss_history = []
            dev_accuracy_history = []
        
        num_batches = len(torch.arange(len(encoded_inputs)).split(self.batch_size))
        for e in range(self.epochs):
            print('Epoch {}/{}'.format(e+1,self.epochs))
            for i, batch in enumerate(batch_iter(
                                            encoded_inputs,
                                            self.batch_size,
                                            pad_idx,
                                            labels=y,
                                            shuffle=True
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
                        encoded_dev_input, y_dev = dev
                        batch = next(batch_iter(
                                            encoded_dev_input,
                                            self.batch_size,
                                            pad_idx,
                                            labels=y_dev,
                                            shuffle=True
                                        ))
                        (input_ids_batch, token_type_ids_batch, attention_mask_batch, labels_batch) =  (x.to(device=device) for x in batch)

                        model.eval()
                        with torch.no_grad():
                            scores = model(
                                input_ids=input_ids_batch,
                                attention_mask=attention_mask_batch,
                                token_type_ids=token_type_ids_batch,
                                labels=labels_batch,
                                output_hidden_states=False,
                                output_attentions=False,
                                return_dict=True
                            )
                            loss = scores['loss']
                            print('Dev loss: {:.5f}'.format(loss.item()))
                            dev_loss_history.append(loss.item())
                            print('Dev accuracy:',end=' ')
                            dev_acc = self.check_accuracy(scores['logits'], labels_batch)
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
        ds = ds.tolist()
        encoded_input = self.tokenizer(
            ds,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            return_attention_mask=True
        )
        encoded_inputs = pd.DataFrame.from_dict({
            'input_ids': encoded_input.input_ids,
            'token_type_ids': encoded_input.token_type_ids,
            'attention_mask': encoded_input.attention_mask
        })
        pad_idx = self.tokenizer.pad_token_id
        labels = torch.zeros(len(ds),dtype=torch.float)

        device = torch.device(self.device_type)
        model = self.model
        model.eval()

        num_batches = len(torch.arange(len(encoded_inputs)).split(self.batch_size))
        y_pred_batches = []
        for batch in tqdm(batch_iter(encoded_inputs,self.batch_size,pad_idx,labels=labels,shuffle=False),total=num_batches):
            input_ids_batch, token_type_ids_batch, attention_mask_batch, _ = (x.to(device=device) for x in batch)
            
            with torch.no_grad():
                scores = model(
                    input_ids=input_ids_batch,
                    attention_mask=attention_mask_batch,
                    token_type_ids=token_type_ids_batch,
                    labels=None,
                    output_hidden_states=False,
                    output_attentions=False,
                    return_dict=True
                )
                y_pred = torch.argmax(scores['logits'],dim=1).cpu().numpy()
                y_pred_batches.append(y_pred)
        
        y_pred = np.hstack(y_pred_batches)

        return y_pred