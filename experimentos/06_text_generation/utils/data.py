import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import pandas as pd


RANDOM_SEED = 61273812
MELISA_PATH = '/'.join(os.getcwd().split('/')[:-2]) + '/datav2/esp/'
# MELISA_PATH = '/'.join(os.getcwd().split('/')) + '/datav2/esp/'


def get_train_dev_idx(N,dev_size=.2,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    if dev_size == 0:
        return rand_idx

    N_train = int(N * (1-dev_size))
    if N_train == N:
        print('Warning: dev_size too small!')
        N_train = N-1
    
    return rand_idx[:N_train], rand_idx[N_train:]


def train_dev_split(df,dev_size,random_state):

    N_data = len(df)
    train_idx, dev_idx = get_train_dev_idx(N_data,dev_size,random_state)
    dev_dataset = df.iloc[dev_idx,:].reset_index(drop=True)
    train_dataset = df.iloc[train_idx,:].reset_index(drop=True)

    return train_dataset, dev_dataset


def load_melisa(split='train'):
    
    path = MELISA_PATH + split + '.csv'
    df = pd.read_csv(path,
                     lineterminator='\n',
                     sep=',',
                     usecols=['review_content','review_title'],
                     dtype={'review_content': str, 'review_title': str})
    
    return df


def load_and_split_melisa(devsize):
    df = load_melisa(split='train')
    df_train, df_dev = train_dev_split(df,devsize,RANDOM_SEED)
    return df_train, df_dev


def load_amazon(split='train'):
    split = 'validation' if split == 'dev' else split
    dataset = load_dataset("amazon_reviews_multi","es")
    df = pd.DataFrame(dataset[split]).loc[:,['review_body','review_title','stars']].sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
    df = df.rename(columns={'review_body':'review_content'})

    return df


def normalize_dataset(ds):

    accents = [
        ('[óòöøôõ]','ó'), ('[áàäåâã]','á'), ('[íìïî]','í'), ('[éèëê]','é'), ('[úùû]','ú'), ('[ç¢]','c'), 
        ('[ÓÒÖØÔÕ]','Ó'), ('[ÁÀÄÅÂÃ]','Á'), ('[ÍÌÏÎ]','Í'), ('[ÉÈËÊ]','É'), ('[ÚÙÛ]','Ù'), ('Ç','C'),
        ('[ý¥]','y'), ('š','s'), ('ß','b'), ('\x08','')
    ]
    for rep, rep_with in accents:
        ds  = ds.str.replace(rep,rep_with,regex=True)

    ds = ds.str.lower()

    return ds