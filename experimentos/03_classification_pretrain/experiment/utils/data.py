import os
import pandas as pd
import numpy as np
from .tokenizers import WordTokenizer
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import xml.etree.ElementTree as ET

RANDOM_SEED = 61273812
MELISA_PATH = '/'.join(os.getcwd().split('/')[:-2]) + '/datav2/esp/'
TASS_PATH = os.path.join(os.getcwd(),'../../other_datasets/tass2012')


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


def load_melisa(split="train",nclasses=2):
    if split in ["train", "dev"]:
        path = os.path.join(MELISA_PATH,"train.csv")
    if split == "test":
        path = os.path.join(MELISA_PATH,"test.csv")
    
    df = pd.read_csv(path,
                     lineterminator='\n',
                     sep=',',
                     usecols=['review_content','review_rate'],
                     dtype={'review_content': str, 'review_rate': int})
    df = df.rename(columns={'review_content':'content', 'review_rate': 'label'})

    if nclasses == 2:
        df = df[df['label'] != 3].reset_index(drop=True)
        df.loc[(df['label'] <= 2),['label']] = 0
        df.loc[(df['label'] >= 4),['label']] = 1
    elif nclasses == 3:
        df.loc[(df['label'] <= 2),['label']] = 0
        df.loc[(df['label'] >= 4),['label']] = 2
        df.loc[df['label'] == 3,['label']] = 1
    else:
        df['label'] = df['label'] - 1

    if split == "train":
        df, _ = train_dev_split(df,0.05,RANDOM_SEED)
    elif split == "dev":
        _, df = train_dev_split(df,0.05,RANDOM_SEED)

    return df

def load_tass(split="train",nclasses=2):
    if split in ["train", "dev"]:
        tree = ET.parse(os.path.join(TASS_PATH,'general-train-tagged.xml'))
    if split == "test":
        tree = ET.parse(os.path.join(TASS_PATH,'general-test-tagged.xml'))

    if nclasses == 2:
        label2num = {'P+': 1, 'P': 1, 'N': 0, 'N+': 0}
    elif nclasses == 3:
        label2num = {'P+': 2, 'P': 2, 'NEU': 1, 'N': 0, 'N+': 0}
    elif nclasses == 5:
        label2num = {'P+': 4, 'P': 3, 'NEU': 2, 'N': 1, 'N+': 0}

    root = tree.getroot()
    dataset = {'content': [], 'label': []}
    for item in root:
        tweet = item[2].text
        label = item[5][0][0].text
        if (label == 'NONE') or (label == 'NEU' and nclasses == 2):
            continue
        num = label2num[label]
        dataset['content'].append(tweet)
        dataset['label'].append(num)
    df = pd.DataFrame.from_dict(dataset)

    if split == "train":
        df, _ = train_dev_split(df,0.1,RANDOM_SEED)
    elif split == "dev":
        _, df = train_dev_split(df,0.1,RANDOM_SEED)

    return df


def load_cine(split="train",nclasses=2):
    CINE_RNDM_SEED = 2374812
    rs = np.random.RandomState(CINE_RNDM_SEED)
    dataset = load_dataset("muchocine")['train']
    df = pd.DataFrame(dataset).loc[:,['review_body','star_rating']]
    df = df.rename(columns={'review_body':'content', 'star_rating': 'label'})
    df = df.loc[df['content'].str.len() > 0,:].reset_index(drop=True)

    if nclasses == 2:
        df = df[df['label'] != 3].reset_index(drop=True)
        df.loc[(df['label'] <= 2),['label']] = 0
        df.loc[(df['label'] >= 4),['label']] = 1
    elif nclasses == 3:
        df.loc[(df['label'] <= 2),['label']] = 0
        df.loc[(df['label'] >= 4),['label']] = 2
        df.loc[df['label'] == 3,['label']] = 1
    else:
        df['label'] = df['label'] - 1

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

def load_amazon(split='train',nclasses=2):
    split = 'validation' if split == 'dev' else split
    dataset = load_dataset("amazon_reviews_multi","es")
    df = pd.DataFrame(dataset[split]).loc[:,['review_body','review_title','stars']].sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
    df = df.rename(columns={'review_body':'content', 'stars': 'label'})
    
    if nclasses == 2:
        df = df[df['label'] != 3].reset_index(drop=True)
        df.loc[(df['label'] <= 2),['label']] = 0
        df.loc[(df['label'] >= 4),['label']] = 1
    elif nclasses == 3:
        df.loc[(df['label'] <= 2),['label']] = 0
        df.loc[(df['label'] >= 4),['label']] = 2
        df.loc[df['label'] == 3,['label']] = 1
    else:
        df['label'] = df['label'] - 1

    return df


class TextDataset(Dataset):
    def __init__(self,inputs_ids,labels):
        self.inputs_ids = inputs_ids
        self.labels = labels

    def __getitem__(self,idx):
        return {
            "input_ids": self.inputs_ids[idx], 
            "label": self.labels[idx]
        }
    
    def __len__(self):
        return len(self.labels)

def pad_batch_sentences(batch,pad_idx=0):
    max_len = max([len(sent["input_ids"]) for sent in batch])
    for sample in batch:
        sample["input_ids"].extend([pad_idx] * (max_len - len(sample["input_ids"])))
    input_ids = torch.LongTensor([sample["input_ids"] for sample in batch])
    attention_mask = (input_ids != pad_idx).float()
    label = torch.LongTensor([sample["label"] for sample in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": label
    }

def load_data_for_pretrain(nclasses,train_batch_size,dev_batch_size,
                    max_tokens,freq_cutoff,max_sent_len,frac=1.):
    RANDOM_SHUFFLE = 16273
    df_train, df_dev = load_melisa("train",nclasses).sample(frac=frac,random_state=RANDOM_SHUFFLE).reset_index(drop=True), load_melisa("dev",nclasses)
    tokenizer = WordTokenizer.from_dataseries(
        df_train["content"],
        freq_cutoff=freq_cutoff,
        max_tokens=max_tokens,
        max_sent_len=max_sent_len,
        pad_token="[PAD]",
        unk_token="[UNK]"
    )
    train_dataloader = DataLoader(
        TextDataset(tokenizer.tokens_to_ids(df_train["content"]),df_train["label"]),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: pad_batch_sentences(batch,pad_idx=tokenizer.vocab[tokenizer.pad_token])
    )
    
    dev_dataloader = DataLoader(
        TextDataset(tokenizer.tokens_to_ids(df_dev["content"]),df_dev["label"]),
        batch_size=dev_batch_size,
        shuffle=False,
        collate_fn=lambda batch: pad_batch_sentences(batch,pad_idx=tokenizer.vocab[tokenizer.pad_token])
    )

    return train_dataloader, dev_dataloader, tokenizer

# if __name__ == "__main__":
#     MELISA_PATH = '/'.join(os.getcwd().split('/')[:-4]) + '/datav2/esp/'

#     train_dataloader, dev_dataloader, tokenizer = load_data_for_pretrain(
#         nclasses=2,train_batch_size=2,dev_batch_size=2,max_tokens=20,
#         freq_cutoff=1,max_sent_len=50)
#     batch = next(iter(dev_dataloader))
#     print(batch)
    
def load_data_for_finetune(nclasses,dataset,train_batch_size,dev_batch_size,
                    max_tokens,freq_cutoff,max_sent_len):
    
    if dataset == "cine":
        df_train = load_cine("train",nclasses)
        df_dev = load_cine("dev",nclasses)
        df_test = load_cine("test",nclasses)
    elif dataset == "tass":
        df_train = load_tass("train",nclasses)
        df_dev = load_tass("dev",nclasses)
        df_test = load_tass("test",nclasses)
    elif dataset == "amazon":
        df_train = load_amazon("train",nclasses)
        df_dev = load_amazon("dev",nclasses)
        df_test = load_amazon("test",nclasses)

    tokenizer = WordTokenizer.from_dataseries(
        df_train["content"],
        freq_cutoff=freq_cutoff,
        max_tokens=max_tokens,
        max_sent_len=max_sent_len,
        pad_token="[PAD]",
        unk_token="[UNK]"
    )
    train_dataloader = DataLoader(
        TextDataset(tokenizer.tokens_to_ids(df_train["content"]),df_train["label"]),
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda batch: pad_batch_sentences(batch,pad_idx=tokenizer.vocab[tokenizer.pad_token])
    )
    
    dev_dataloader = DataLoader(
        TextDataset(tokenizer.tokens_to_ids(df_dev["content"]),df_dev["label"]),
        batch_size=dev_batch_size,
        shuffle=False,
        collate_fn=lambda batch: pad_batch_sentences(batch,pad_idx=tokenizer.vocab[tokenizer.pad_token])
    )

    test_dataloader = DataLoader(
        TextDataset(tokenizer.tokens_to_ids(df_test["content"]),df_test["label"]),
        batch_size=dev_batch_size,
        shuffle=False,
        collate_fn=lambda batch: pad_batch_sentences(batch,pad_idx=tokenizer.vocab[tokenizer.pad_token])
    )

    return train_dataloader, dev_dataloader, test_dataloader, tokenizer