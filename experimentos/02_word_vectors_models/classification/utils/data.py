import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import xml.etree.ElementTree as ET
import pandas as pd


RANDOM_SEED = 61273812
MELISA_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/datav2/esp/'
TASS_PATH = os.path.join(os.getcwd(),'../../../other_datasets/tass2012')


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

def load_melisa(split='train',nclasses=2):
    
    path = MELISA_PATH + split + '.csv'
    df = pd.read_csv(path,
                     lineterminator='\n',
                     sep=',',
                     usecols=['review_content','review_rate'],
                     dtype={'review_content': str, 'review_rate': int})
    
    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1.
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2.
        df.loc[df['review_rate'] == 3,['review_rate']] = 1.
    else:
        df['review_rate'] = df['review_rate'] - 1

    return df

def load_and_split_melisa(nclasses,devsize):
    df = load_melisa(
            split='train',
            nclasses=nclasses
        )
    df_train, df_dev = train_dev_split(df,devsize,RANDOM_SEED)
    return df_train, df_dev


def load_amazon(split='train',nclasses=2):
    split = 'validation' if split == 'dev' else split
    dataset = load_dataset("amazon_reviews_multi","es")
    df = pd.DataFrame(dataset[split]).loc[:,['review_body','review_title','stars']].sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
    df = df.rename(columns={'review_body':'review_content', 'stars': 'review_rate'})
    
    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1.
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2.
        df.loc[df['review_rate'] == 3,['review_rate']] = 1.
    else:
        df['review_rate'] = df['review_rate'] - 1

    return df

def load_tass(split='train', nclasses=2):
    tree = ET.parse(os.path.join(TASS_PATH,'general-' + split + '-tagged.xml'))
    root = tree.getroot()
    if nclasses == 2:
        label2num = {'P+': 1, 'P': 1, 'N': 0, 'N+': 0}
    elif nclasses == 3:
        label2num = {'P+': 2, 'P': 2, 'NEU': 1, 'N': 0, 'N+': 0}
    elif nclasses == 5:
        label2num = {'P+': 4, 'P': 3, 'NEU': 2, 'N': 1, 'N+': 0}
    
    dataset = {'tweet': [], 'label': []}
    for item in root:
        tweet = item[2].text
        label = item[5][0][0].text
        if (label == 'NONE') or (label == 'NEU' and nclasses == 2):
            continue
        num = label2num[label]
        dataset['tweet'].append(tweet)
        dataset['label'].append(num)

    dataset = pd.DataFrame.from_dict(dataset)
    return dataset

def load_and_split_tass(nclasses,devsize):
    df = load_tass(
            split='train',
            nclasses=nclasses
        )
    df_train, df_dev = train_dev_split(df,devsize,RANDOM_SEED)
    return df_train, df_dev


def load_cine(nclasses,split='train'):
    CINE_RNDM_SEED = 2374812
    rs = np.random.RandomState(CINE_RNDM_SEED)
    dataset = load_dataset("muchocine")['train']
    print(dataset)
    df = pd.DataFrame(dataset).loc[:,['review_body','review_summary','star_rating']]
    df = df.rename(columns={'review_body':'review_content', 'star_rating': 'review_rate', 'review_summary': 'review_title'})
    df = df.loc[df['review_content'].str.len() > 0,:].reset_index(drop=True)

    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1.
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2.
        df.loc[df['review_rate'] == 3,['review_rate']] = 1.
    else:
        df['review_rate'] = df['review_rate'] - 1


    N = len(df) 
    N_train, N_dev = int(N*0.85), int(N*0.05)
    indices = rs.permutation(N)

    if split == 'train':
        df_train = df.iloc[indices[:N_train],:].reset_index(drop=True)
        return df_train
    elif split == 'dev':
        df_dev = df.iloc[indices[N_train:N_train+N_dev],:].reset_index(drop=True)
        return df_dev
    elif split == 'test':
        df_test = df.iloc[indices[N_train+N_dev:],:].reset_index(drop=True)
        return df_test


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