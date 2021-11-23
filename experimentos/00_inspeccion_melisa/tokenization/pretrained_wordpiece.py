import argparse
import os
from utils import load_data
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from transformers import BertTokenizer
from collections import Counter

RANDOM_SEED = 61273812
DEV_SIZE = 0.1
TRAIN_DATA_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/datav2/esp/train.csv'
MODEL_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/pretrained_models/'


parser = argparse.ArgumentParser()
parser.add_argument('--lower', action='store_true', required=False, default=False)
parser.add_argument('--process_accents', action='store_true', required=False, default=False)


def parse_args():
    args = parser.parse_args()
    args = dict(
        process_accents=args.process_accents,
        lower=args.lower
    )
    return args


def save_results(ds_tokenized,unk_counts):

    now = datetime.now()
    title = now.strftime("%Y-%m-%d-%H-%M-%S")

    with open('results/{}-words.tokenized'.format(title),'w') as f:
        f.write('UNKs found / total words: {}/{} ({:.2f}%)\n'.format(sum(unk_counts),ds_tokenized.str.len().sum(),
                                                            sum(unk_counts)/ds_tokenized.str.len().sum()*100))
        f.write('% UNKs per sent (avg): {:.2f}%\n\n'.format((unk_counts / ds_tokenized.str.len()).mean()*100))
        for sent in ds_tokenized:
            f.write("{}\n".format(sent))

def normalize(corpus,process_accents,lower):
    if process_accents:
        accents = [
            ('[óòöøôõ]','ó'), ('[áàäåâã]','á'), ('[íìïî]','í'), 
            ('[éèëê]','é'), ('[úùû]','ú'), ('[ç¢]','c'), 
            ('[ÓÒÖØÔÕ]','Ó'), ('[ÁÀÄÅÂÃ]','Á'), ('[ÍÌÏÎ]','Í'), 
            ('[ÉÈËÊ]','É'), ('[ÚÙÛ]','Ù'), ('Ç','C'),
            ('[ý¥]','y'), ('š','s'), ('ß','b'), ('\x08','')
        ]
        for rep, rep_with in accents:
            corpus  = corpus.str.replace(rep,rep_with,regex=True)
    if lower:
        corpus = corpus.str.lower()
    
    return corpus

def tokenize(tokenizer,corpus):
    unk_token = tokenizer.unk_token
    unks_counts = []
    tokenized_corpus = []
    for sent in tqdm(corpus):
        tokenized_sent = tokenizer.tokenize(sent)
        counts = Counter(tokenized_sent)
        uc = counts.get(unk_token,0)
        unks_counts.append(uc)
        tokenized_corpus.append(tokenized_sent)

    return pd.Series(tokenized_corpus), pd.Series(unks_counts)


def main():
    args = parse_args()
    _, ds_dev = load_data(TRAIN_DATA_PATH,DEV_SIZE,RANDOM_SEED)
    do_lower_case = args['lower']
    path = MODEL_PATH + 'beto_uncased' if do_lower_case else MODEL_PATH + 'beto_cased'
    tokenizer = BertTokenizer.from_pretrained(path,do_lower_case=do_lower_case)
    ds_dev = normalize(ds_dev,args['process_accents'],args['lower'])
    ds_dev_tokenized, unk_counts = tokenize(tokenizer,ds_dev)

    save_results(ds_dev_tokenized,unk_counts)



if __name__ == '__main__':
    main()