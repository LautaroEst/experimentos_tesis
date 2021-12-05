from datetime import datetime
import os
import pickle
from time import time
import numpy as np
from torch import optim
import utils
from utils.data import load_melisa, load_amazon, load_and_split_melisa, normalize_dataset
from utils.io import parse_args
from models import init_lstm_model, WordTokenizer


import pandas as pd
import torch


def load_dataset(dataset,devsize):
    print('Loading dataset...')
    if dataset == 'melisa':
            df_train, df_dev = load_and_split_melisa(devsize)
    elif dataset == 'amazon':
            df_train = load_amazon(split='train')
            df_dev = load_amazon(split='dev')
    else:
        raise NameError('Dataset not supported')

    print('Normalizing dataset...')
    #n = 1000
    ds_src_train = normalize_dataset(df_train['review_content'])#.iloc[:n])
    ds_target_train = normalize_dataset(df_train['review_title'])#.iloc[:n])
    ds_src_dev = normalize_dataset(df_dev['review_content'])#.iloc[:n])
    ds_target_dev = normalize_dataset(df_dev['review_title'])#.iloc[:n])
    data = {
        'src_train': ds_src_train, 
        'tgt_train': ds_target_train, 
        'src_dev': ds_src_dev, 
        'tgt_dev': ds_target_dev
    }
    return data


def init_model(model_kwargs,src_tokenizer_kwargs,tgt_tokenizer_kwargs):

    model_name = model_kwargs.pop('name')
    src_tokenizer = init_tokenizer(**src_tokenizer_kwargs)
    tgt_tokenizer = init_tokenizer(**tgt_tokenizer_kwargs)
    
    if model_name == "lstm":
        model = init_lstm_model(src_tokenizer,tgt_tokenizer,**model_kwargs)
    elif model_name == "transformer":
        pass
    else:
        raise NameError("Model not supported")

    return model, src_tokenizer, tgt_tokenizer


def init_tokenizer(**tokenizer_kwargs):

    tokenizer_name = tokenizer_kwargs.pop('name')

    if tokenizer_name == "word_tokenizer":
        tokenizer = WordTokenizer.from_dataseries(**tokenizer_kwargs)
    elif tokenizer_name == "transformer":
        pass
        # tokenizer = BertTokenizer.from_pretrained(src_path,**tokenizer_kwargs)
    else:
        raise NameError("Model not supported")
    
    return tokenizer


def batch_iter(src_sents,tgt_sents,src_tokenizer,tgt_tokenizer,batch_size,shuffle=True):

    df = pd.concat((src_sents,tgt_sents),keys=['src','tgt'],axis=1)
    N = len(df)

    if shuffle:
        indices_batches = torch.randperm(N).split(batch_size)
    else:
        indices_batches = torch.arange(N).split(batch_size)

    for i, indices in enumerate(indices_batches):
        batch = df.iloc[indices,:]
        src_encoded_input = src_tokenizer(batch['src'])
        tgt_encoded_input = tgt_tokenizer(batch['tgt'])

        _, indices = src_encoded_input['attention_mask'].sum(dim=1).sort(descending=True)
        src_encoded_input['input_ids'] = src_encoded_input['input_ids'][indices,:]
        src_encoded_input['attention_mask'] = src_encoded_input['attention_mask'][indices,:]
        tgt_encoded_input['input_ids'] = tgt_encoded_input['input_ids'][indices,:]
        tgt_encoded_input['attention_mask'] = tgt_encoded_input['attention_mask'][indices,:]
        
        yield i, (src_encoded_input, tgt_encoded_input)


def train_model(
        model,
        data,
        src_tokenizer,
        tgt_tokenizer,
        results_dir,
        **kwargs
    ):

    max_epochs = kwargs.pop('max_epochs')
    lr = kwargs.pop('learning_rate')
    train_batch_size = kwargs.pop('train_batch_size')
    dev_batch_size = kwargs.pop('dev_batch_size')
    train_eval_every = kwargs.pop('train_eval_every')
    dev_eval_every = kwargs.pop('dev_eval_every')
    
    optimizer = optim.Adam(model.parameters(),lr=lr)

    train_ppl_history = []
    dev_ppl_history = []
    
    num_batches = len(torch.arange(len(data['src_train'])).split(train_batch_size))
    cum_train_ppl = cum_num_examples = total_words_to_predict = e = 0
    start_time = time()
    while e < max_epochs:
        for i, batch in batch_iter(
                            data['src_train'], data['tgt_train'],
                            src_tokenizer,tgt_tokenizer,
                            train_batch_size,
                            shuffle=True
                        ):

            src_encoded_input, tgt_encoded_input = batch

            optimizer.zero_grad()
            _, loss = model(src_encoded_input,tgt_encoded_input)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            num_examples = src_encoded_input['input_ids'].size(0)
            cum_train_ppl += loss_val
            cum_num_examples += num_examples
            total_words_to_predict += src_encoded_input['attention_mask'][:,1:].sum().item()

            if (e * num_batches + i) % train_eval_every == 0:
                print('Batch {}/{}. Epoch {}/{}'.format(i,num_batches,e+1,max_epochs))
                avg_ppl = np.exp(cum_train_ppl / total_words_to_predict)
                print('Train avg. ppl: {:.5f}'.format(avg_ppl))
                train_ppl_history.append(avg_ppl)
                cum_train_ppl = cum_num_examples = total_words_to_predict = 0

            if (e * num_batches + i) % dev_eval_every == 0:
                print('Dev evaluation...')
                dev_ppl = evaluate_ppl(model,data['src_dev'],data['tgt_dev'],src_tokenizer,tgt_tokenizer,dev_batch_size)
                print('Dev avg. ppl: {:.5f}'.format(dev_ppl))
                dev_ppl_history.append(dev_ppl)
                
        e += 1

    end_time = time()
    print("Duración del entrenamiento: {:.2f}".format((end_time-start_time) / 3600))

    with open(os.path.join(results_dir,'train_ppl_history.pkl'), "wb") as f:
        pickle.dump(train_ppl_history,f)
    with open(os.path.join(results_dir,'dev_ppl_history.pkl'), "wb") as f:
        pickle.dump(dev_ppl_history,f)

    model.save(os.path.join(results_dir,'model.pkl'))
    torch.save(model.state_dict(),os.path.join(results_dir,'model_state_dict.pkl'))
    torch.save(optimizer.state_dict(),os.path.join(results_dir,'optimizer_state_dict.pkl'))



def evaluate_ppl(model,src_dev,tgt_dev,src_tokenizer,tgt_tokenizer,batch_size=32):
    
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for i, (src_encoded_input, tgt_encoded_input) in batch_iter(
                            src_dev, tgt_dev,
                            src_tokenizer,tgt_tokenizer,
                            batch_size,
                            shuffle=False
                        ):
            _, loss = model(src_encoded_input,tgt_encoded_input)

            cum_loss += loss.item()
            tgt_word_num_to_predict = tgt_encoded_input['attention_mask'][:,1:].sum().item() # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def main():

    # Read args
    args = parse_args()
        
    # Dataset loading:
    dataset_args = args.pop('dataset_args')
    data = load_dataset(**dataset_args)

    # Model Initialization
    model_kwargs = args.pop('model_kwargs')
    src_tokenizer_kwargs = args.pop('src_tokenizer_kwargs')
    tgt_tokenizer_kwargs = args.pop('tgt_tokenizer_kwargs')
    
    if (src_tokenizer_kwargs['name'] == 'word_tokenizer') or \
        (tgt_tokenizer_kwargs['name'] == 'word_tokenizer'):
        ds = pd.concat([data['src_train'],data['tgt_train']],ignore_index=True)
    if src_tokenizer_kwargs['name'] == 'word_tokenizer':
        src_tokenizer_kwargs['ds'] = ds
    if tgt_tokenizer_kwargs['name'] == 'word_tokenizer':
        tgt_tokenizer_kwargs['ds'] = ds

    model, src_tokenizer, tgt_tokenizer = init_model(
        model_kwargs,
        src_tokenizer_kwargs,
        tgt_tokenizer_kwargs
    )

    # Training
    train_model(
        model=model,
        data=data,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        results_dir=args['results_dir'],
        **args['train_kwargs']
    )
    



if __name__ == '__main__':
    main()