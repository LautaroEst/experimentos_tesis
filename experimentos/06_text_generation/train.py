from datetime import datetime
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
    n = 1000
    ds_src_train = normalize_dataset(df_train['review_content'].iloc[:n])
    ds_target_train = normalize_dataset(df_train['review_title'].iloc[:n])
    ds_src_dev = normalize_dataset(df_dev['review_content'].iloc[:n])
    ds_target_dev = normalize_dataset(df_dev['review_title'].iloc[:n])
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

    return model


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


def batch_iter(data,batch_size,shuffle=True):

    src_sents, tgt_sents = data
    N = len(src_sents)

    if shuffle:
        indices_batches = torch.randperm(N).split(batch_size)
    else:
        indices_batches = torch.arange(N).split(batch_size)

    for i, idx in enumerate(indices_batches):
        src_sents_batch = src_sents.iloc[idx].reset_index(drop=True)
        tgt_sents_batch = tgt_sents.iloc[idx].reset_index(drop=True)

        yield i, (src_sents_batch, tgt_sents_batch)


def train_model(
        model,
        data,
        results_dir,
        **kwargs
    ):

    max_epochs = kwargs.pop('max_epochs')
    lr = kwargs.pop('learning_rate')
    train_batch_size = kwargs.pop('train_batch_size')
    dev_batch_size = kwargs.pop('dev_batch_size')
    
    optimizer = optim.Adam(model.parameters(),lr=lr)
    
    e = 0
    while e < max_epochs:
        for i, batch in batch_iter(
                            (data['src_train'], data['tgt_train']),
                            train_batch_size,
                            shuffle=True
                        ):

            src_sents_batch, tgt_sents_batch = batch

            optimizer.zero_grad()

            pred_sent = model(src_sents_batch,tgt_sents_batch)
            loss = pred_sent.sum()
            print("loss val: {:.2f}".format(loss.item()))
            loss.backward()
            optimizer.step()



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
    
    if src_tokenizer_kwargs['name'] == 'word_tokenizer':
        src_tokenizer_kwargs['ds'] = data['src_train']
    if tgt_tokenizer_kwargs['name'] == 'word_tokenizer':
        tgt_tokenizer_kwargs['ds'] = data['tgt_train']

    model = init_model(
        model_kwargs,
        src_tokenizer_kwargs,
        tgt_tokenizer_kwargs
    )

    # Training
    train_model(
        model=model,
        data=data,
        results_dir=args['results_dir'],
        **args['train_kwargs']
    )
    



if __name__ == '__main__':
    main()