import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

import sys
sys.path.append('/home/lestien/Documents/Trabajos 2021/melisa/experimentos/')
import utils as ut
import pickle

DATA_PATH = '/home/lestien/Documents/Trabajos 2021/melisa/datav2/esp/'
BERT_PATH = '/home/lestien/Documents/Trabajos 2021/melisa/experimentos/05_experiment_beto/bert/'


def normalize_dataset(df):
    # Pasamos a minúscula todo
    df['review_content'] = df['review_content'].str.lower()
    # Sacamos todos los acentos
    for rep, rep_with in [('[óòÓöøôõ]','o'), ('[áàÁäåâãÄ]','a'), ('[íìÍïîÏ]','i'), 
                            ('[éèÉëêÈ]','e'), ('[úüÚùûÜ]','u'), ('[ç¢Ç]','c'), 
                            ('[ý¥]','y'),('š','s'),('ß','b'),('\x08','')]:
        df['review_content']  = df['review_content'].str.replace(rep,rep_with,regex=True)
    return df




class BertClassifier(object):

    def __init__(self,model_path,n_iters,batch_size,learning_rate,
            weight_decay,dropout_last_layer,n_classes,device='cuda:1'):
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

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
            classifier_dropout=dropout_last_layer,
            num_labels=n_classes
        )
        self.model = BertForSequenceClassification.from_pretrained(model_path,config=config)
        self.n_classes = n_classes
        self.max_len = config_json['max_position_embeddings']
        self.device = device

    def fit(self,encoded_input,labels):
        device = torch.device(self.device)
        
        labels = torch.from_numpy(labels).type(torch.long)
        dataset = TensorDataset(encoded_input['input_ids'],
                                encoded_input['attention_mask'],
                                encoded_input['token_type_ids'],
                                labels)
        dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        model = self.model
        model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=self.learning_rate,
                            weight_decay=self.weight_decay)
        model.train()
        loss_history = []
        print('Training...')
        for e in range(self.n_iters):
            for i, batch in enumerate(dataloader):
                input_ids, attention_mask, token_type_ids, y = (t.to(device=device) for t in batch)

                # Forward
                output = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=y,
                            output_hidden_states=False,
                            output_attentions=False,
                            return_dict=True)
                loss = output['loss']
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Update
                optimizer.step()

                loss_history.append(loss.item())

            print('epoch {}/{}. Loss: {:.4f}'.format(e+1,self.n_iters,loss.item()))

        model.eval()
        self.model = model
        return loss_history

    def predict(self,encoded_input):
        model = self.model
        model.eval()
        device = torch.device(self.device)

        activation = lambda logprobs: logprobs.argmax(dim=1).type(torch.int).detach().view(-1).cpu().tolist()
        dataset = TensorDataset(encoded_input['input_ids'],
                                encoded_input['attention_mask'],
                                encoded_input['token_type_ids'])
        dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=False)

        predicted_labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, token_type_ids = (t.to(device=device) for t in batch)

                # Forward
                output = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=False,
                            output_attentions=False,
                            return_dict=True)
                logprobs = output['logits']
                y_pred = activation(logprobs)
                predicted_labels.extend(y_pred)
        
        return np.array(predicted_labels)

def main():
    df = ut.load_data(DATA_PATH,'train',nclasses=5).loc[:,['review_content','review_rate']]
    #df = df.sample(frac=0.001,random_state=27381)

    df = normalize_dataset(df)
    df_train, df_dev = ut.train_dev_split(df,dev_size=0.1,random_state=2376482)
    y_train, y_dev = df_train['review_rate'].values-1, df_dev['review_rate'].values-1

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH,do_lower_case=False)

    with open(BERT_PATH + 'config.json', 'r') as f:
        max_len = json.load(f)['max_position_embeddings']

    print('Encoding train input...')
    encoded_train_input = tokenizer(df_train['review_content'].tolist(),
                                add_special_tokens=True,
                                padding=True,
                                truncation=True,
                                max_length=max_len,
                                return_tensors='pt')

    print('Encoding dev input...')
    encoded_dev_input = tokenizer(df_dev['review_content'].tolist(),
                                add_special_tokens=True,
                                padding=True,
                                truncation=True,
                                max_length=max_len,
                                return_tensors='pt')

    dropout_probs = [0., 0.1, 0.2]
    learning_rates = [1e-4, 5e-5, 1e-5]
    weight_decays = [0.01, 0.1]#[0., 0.001, 0.01, 0.1]
    results = {dp: {lr: {wd: {} for wd in weight_decays} for lr in learning_rates} for dp in dropout_probs}
    for dp in dropout_probs:
        for lr in learning_rates:
            for wd in weight_decays:
                clf = BertClassifier(model_path=BERT_PATH,n_iters=5,
                batch_size=8,learning_rate=lr,weight_decay=wd,
                dropout_last_layer=dp,n_classes=5,device='cuda:1')
    
                loss_history = clf.fit(encoded_train_input,y_train)

                y_pred = clf.predict(encoded_dev_input)

                print('Training: dropout {}, learning rate {}, weight decay {}'.format(dp,lr,wd))
                print('Accuracy: {:.2f}%'.format( (y_dev == y_pred).mean()*100 ))
                results[dp][lr][wd]['y'] = y_dev.copy()
                results[dp][lr][wd]['y_pred'] = y_pred.copy()
                results[dp][lr][wd]['loss_history'] = loss_history

                with open('./results_hyper_search/checkpoint_{}_{}_{}.pkl'.format(dp,lr,wd),'wb') as f:
                    pickle.dump(results,f)

                del clf

if __name__ == '__main__':
    main()

