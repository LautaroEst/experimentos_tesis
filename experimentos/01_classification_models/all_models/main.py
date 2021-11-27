from datetime import datetime
from models import *
import sys
from utils import *
import json
import os


MODELS_PATH = os.getcwd() + "/models.json"
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
            pass
        else:
            pass
    elif dataset == 'muchocine':

        if split == 'dev':
            pass
        else:
            pass
    elif dataset == 'tass':

        if split == 'dev':
            pass
        else:
            pass
            
    return data


def main():
    classifiers = load_models()
    args = parse_args(classifiers)

    print("Initializing classifier...")
    model, eval_every = init_clf(args)
    
    print("Loading data...")
    ds_train, y_train, ds_devtest, y_devtest = load_dataset(**args['dataset_args'])

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