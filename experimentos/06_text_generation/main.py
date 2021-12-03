from torch import optim
import utils
from utils.data import load_melisa, load_amazon, load_and_split_melisa, normalize_dataset
from utils.io import parse_args
from models import init_lstm_model

import pandas as pd
import torch


def load_dataset(dataset,devsize,split):
    print('Loading dataset...')
    if dataset == 'melisa':
        if split == 'dev':
            df_train, df_devtest = load_and_split_melisa(devsize)
        else:
            df_train = load_melisa('train')
            df_devtest = load_melisa('test')

    elif dataset == 'amazon':
        if split == 'dev':
            df_train = load_amazon(split='train')
            df_devtest = load_amazon(split='dev')
        else:
            df_train = pd.concat([
                load_amazon(split='train'),
                load_amazon(split='dev')
                ],ignore_index=True
            )
            df_devtest = load_amazon(split='test')
    else:
        raise NameError('Dataset not supported')

    print('Normalizing dataset...')
    n = 1000
    ds_src_train = normalize_dataset(df_train['review_content'].iloc[:n])
    ds_target_train = normalize_dataset(df_train['review_title'].iloc[:n])
    ds_src_devtest = normalize_dataset(df_devtest['review_content'].iloc[:n])
    ds_target_devtest = normalize_dataset(df_devtest['review_title'].iloc[:n])
    data = (ds_src_train, ds_target_train, ds_src_devtest, ds_target_devtest)
    return data


def init_model(model_name,model_args):
    if model_name == "lstm":
        model = init_lstm_model(**model_args)
    elif model_name == "transformer":
        pass
    else:
        raise NameError("Model not supported")
    
    lr = model_args['lr']
    optimizer = optim.Adam(model.parameters(),lr=lr)

    return model, optimizer


def batch_iter(data,batch_size,shuffle=True):

    src_sents, tgt_sents = data
    N = len(src_sents)

    if shuffle:
        indices_batches = torch.randperm(N).split(batch_size)
    else:
        indices_batches = torch.arange(N).split(batch_size)

    for i, idx in enumerate(indices_batches):
        src_sents_batch = src_sents[idx]
        tgt_sents_batch = tgt_sents[idx]

        yield i, (src_sents_batch, tgt_sents_batch)


def train_model(model,data,optimizer,eval_every):

    num_epochs = 2
    lr = 1e-3
    batch_size = 1
    
    
    for e in range(num_epochs):
        for i, batch in batch_iter(data,batch_size,shuffle=True):
            src_sents_batch, tgt_sents_batch = batch

            optimizer.zero_grad()

            pred_sent = model(src_sents_batch,tgt_sents_batch)
            loss = pred_sent.sum()
            loss.backward()
            optimizer.step()



def main():
    # Argument parsing:
    dataset_args, model_name, model_args, eval_every = parse_args()
    
    # Dataset loading:
    data = load_dataset(**dataset_args)
    data = (
        torch.tensor([[1,9,8,5,0,0,0],[4,5,2,1,1,1,0]]).long(),
        torch.tensor([[8,5,0,0,0],[2,1,1,1,0]]).long()
    )

    # Model Initialization
    model, optimizer = init_model(model_name,model_args)

    # Training
    train_model(model,data,optimizer,eval_every)
    



if __name__ == '__main__':
    main()