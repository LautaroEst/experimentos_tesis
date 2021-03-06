from datetime import datetime
from models import *
import sys
from utils import *
import json
import os
import pandas as pd
import numpy as np


MODELS_PATH = os.getcwd() + "/models.json"
# MODELS_PATH = os.getcwd() + "/experimentos/01_classification_models/all_models/models.json"
# MODELS_PATH = os.getcwd() + "/experimentos/01_classification_models/all_models/models.json"
# MODELS_PATH = "/mnt/disco.mafalda/home/lestien/Documents/Trabajos 2021/melisa/experimentos/01_classification_models/all_models/models.json"

def load_models():
    with open(MODELS_PATH,'r') as f:
        classifiers = json.load(f)
    
    for key in classifiers.keys():
        classifiers[key]["clf"] = getattr(sys.modules[__name__],classifiers[key]["clf"])

    return classifiers


def init_clf(args):

    clf_cls, clf_args = args['clf_cls'], args['clf_args']
    model = clf_cls(**clf_args)
    eval_every = args['eval_every']

    return model, eval_every


def load_dataset(nclasses,dataset,devsize,split):
    if dataset == 'melisa':
        if split == 'dev':
            df_train, df_devtest = load_and_split_melisa(nclasses,devsize)
            ds_train = normalize_dataset(df_train['review_content'])
        else:
            df_train = load_melisa('train',nclasses)
            df_devtest = load_melisa('test',nclasses)

        ds_train = normalize_dataset(df_train['review_content'])
        y_train = df_train['review_rate'].values
        ds_devtest = normalize_dataset(df_devtest['review_content'])
        y_devtest = df_devtest['review_rate'].values
        data = (ds_train, y_train, ds_devtest, y_devtest)

    elif dataset == 'amazon':
        if split == 'dev':
            df_train = load_amazon(split='train',nclasses=nclasses)
            df_devtest = load_amazon(split='dev',nclasses=nclasses)
        else:
            df_train = pd.concat(
                [load_amazon(split='train',nclasses=nclasses),
                load_amazon(split='dev',nclasses=nclasses)],
                ignore_index=True
            )
            df_devtest = load_amazon(split='test',nclasses=nclasses)

        ds_train = normalize_dataset(df_train['review_content'])
        y_train = df_train['review_rate'].values
        ds_devtest = normalize_dataset(df_devtest['review_content'])
        y_devtest = df_devtest['review_rate'].values
        data = (ds_train, y_train, ds_devtest, y_devtest)

    elif dataset == 'tass':
        if split == 'dev':
            df_train, df_devtest = load_and_split_tass(nclasses,devsize)
            ds_train = normalize_dataset(df_train['tweet'])
        else:
            df_train = load_tass('train',nclasses)
            df_devtest = load_tass('test',nclasses)

        ds_train = normalize_dataset(df_train['tweet'])
        y_train = df_train['label'].values
        ds_devtest = normalize_dataset(df_devtest['tweet'])
        y_devtest = df_devtest['label'].values
        data = (ds_train, y_train, ds_devtest, y_devtest)

    elif dataset == 'cine':
        if split == 'dev':
            df_train = load_cine(split='train',nclasses=nclasses)
            df_devtest = load_cine(split='dev',nclasses=nclasses)
        else:
            df_train = pd.concat(
                [load_cine(split='train',nclasses=nclasses),
                load_cine(split='dev',nclasses=nclasses)],
                ignore_index=True
            )
            df_devtest = load_cine(split='test',nclasses=nclasses)

        def limit_len(ds):
            ds = ds.apply(lambda s: s[:2000])
            return ds

        ds_train = normalize_dataset(df_train['review_content'])
        ds_train = limit_len(ds_train)
        y_train = df_train['review_rate'].values
        ds_devtest = normalize_dataset(df_devtest['review_content'])
        ds_devtest = limit_len(ds_devtest)
        y_devtest = df_devtest['review_rate'].values
        data = (ds_train, y_train, ds_devtest, y_devtest)
            
    return data


def main():
    classifiers = load_models()
    args = parse_args(classifiers)

    print("Initializing classifier...")
    model, eval_every = init_clf(args)

    print("Loading data...")
    ds_train, y_train, ds_devtest, y_devtest = load_dataset(**args['dataset_args'])
    # ds_train, y_train, ds_devtest, y_devtest = ds_train[:1000], y_train[:1000], ds_devtest[:1000], y_devtest[:1000]

    # counts, bins = np.histogram(ds_train.str.len(),bins=10)
    # print(pd.Series(counts,index=bins[:-1]))
    # print(pd.Series(y_train).value_counts())
    # counts, bins = np.histogram(ds_devtest.str.len(),bins=10)
    # print(pd.Series(counts,index=bins[:-1]))
    # print(pd.Series(y_devtest).value_counts())

    # Model training:
    print('Training...')
    history = model.train(
                ds_train,y_train,
                eval_every=eval_every,
                dev=(ds_devtest,y_devtest)
            )

    # Model evaluation:
    print('Evaluating results...')
    y_train_predict = model.predict(ds_train)
    y_devtest_predict = model.predict(ds_devtest)
    show_results(y_train_predict,y_train,y_devtest_predict,y_devtest,args,history)


if __name__ == '__main__':
    main()