import argparse
import re
import os
from utils import load_data
from collections import defaultdict, Counter
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from nltk import word_tokenize, TweetTokenizer



RANDOM_SEED = 61273812
DEV_SIZE = 0.1
TRAIN_DATA_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/datav2/esp/train.csv'


parser = argparse.ArgumentParser()
parser.add_argument('--lower', action='store_true', required=False, default=False)
parser.add_argument('--process_accents', action='store_true', required=False, default=False)
parser.add_argument('--max_tokens', type=int, required=True)
parser.add_argument('--min_freq', type=int, required=True)
parser.add_argument('--pattern', type=str, required=True)


def parse_args():
    args = parser.parse_args()
    args = dict(
        max_tokens=args.max_tokens,
        min_freq=args.min_freq,
        pattern=args.pattern,
        process_accents=args.process_accents,
        lower=args.lower
    )
    return args


class WordTokenizer(object):
    
    def __init__(
            self,
            max_tokens=10000,
            min_freq=0,
            pattern=r'\w+',
            lower=True,
            process_accents=True,
            unk_token='[UNK]'
        ):

        if pattern == 'nltk':
            self.pre_tokenize = lambda sent: word_tokenize(sent,'spanish')
        elif pattern == 'tweet':
            self.tweet_tokenizer = TweetTokenizer()
            self.pre_tokenize = lambda sent: self.tweet_tokenizer.tokenize(sent)
        else:
            self.pattern = re.compile(pattern)
            self.pre_tokenize = lambda sent: self.pattern.findall(sent)

        self.max_tokens = max_tokens
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.lower = lower
        self.process_accents = process_accents

    def normalize(self,corpus):
        if self.process_accents:
            accents = [
                ('[óòöøôõ]','ó'), ('[áàäåâã]','á'), ('[íìïî]','í'), 
                ('[éèëê]','é'), ('[úùû]','ú'), ('[ç¢]','c'), 
                ('[ÓÒÖØÔÕ]','Ó'), ('[ÁÀÄÅÂÃ]','Á'), ('[ÍÌÏÎ]','Í'), 
                ('[ÉÈËÊ]','É'), ('[ÚÙÛ]','Ù'), ('Ç','C'),
                ('[ý¥]','y'), ('š','s'), ('ß','b'), ('\x08','')
            ]
            for rep, rep_with in accents:
                corpus  = corpus.str.replace(rep,rep_with,regex=True)
        if self.lower:
            corpus = corpus.str.lower()
        
        return corpus

    def train(self,corpus):
        corpus = self.normalize(corpus)

        counts = defaultdict(lambda: 0)
        for sent in tqdm(corpus):
            tokens_counts = Counter(self.pre_tokenize(sent))
            for tk, c in tokens_counts.items():
                counts[tk] += c
        valid_words = {tk: freq for tk, freq in counts.items() if freq >= self.min_freq}
        valid_words = sorted(valid_words.keys(),key=lambda tk: valid_words[tk],reverse=True)[:self.max_tokens-1]
        self.vocab = {tk: idx for idx, tk in enumerate(valid_words,1)}
        self.vocab[self.unk_token] = 0
        return {tk: counts[tk] for tk in valid_words}


    def tokenize(self,corpus):
        vocab = self.vocab
        unk_token = self.unk_token
        
        corpus = self.normalize(corpus)
        unks_counts = []
        tokenized_corpus = []
        for sent in tqdm(corpus):
            tokenized_sent = self.pre_tokenize(sent)
            uc = 0
            for i, tk in enumerate(tokenized_sent):
                if tk not in vocab:
                    tokenized_sent[i] = unk_token
                    uc += 1
            unks_counts.append(uc)
            tokenized_corpus.append(tokenized_sent)

        return pd.Series(tokenized_corpus), pd.Series(unks_counts)


def save_results(freqs,ds_tokenized,unk_counts):

    now = datetime.now()
    title = now.strftime("%Y-%m-%d-%H-%M-%S")

    with open('results/{}-words.freqs'.format(title),'w') as f:
        f.write('Vocab size: {}\n\n'.format(len(freqs)))
        f.write('Word\t\t\tCount\n\n')
        for tk, freq in freqs.items():
            f.write("{}\t\t\t{}\n".format(tk,freq))

    with open('results/{}-words.tokenized'.format(title),'w') as f:
        f.write('UNKs found / total words: {}/{} ({:.2f}%)\n'.format(sum(unk_counts),ds_tokenized.str.len().sum(),
                                                            sum(unk_counts)/ds_tokenized.str.len().sum()*100))
        f.write('% UNKs per sent (avg): {:.2f}%\n\n'.format((unk_counts / ds_tokenized.str.len()).mean()*100))
        for sent in ds_tokenized:
            f.write("{}\n".format(sent))


def main():
    args = parse_args()
    ds_train, ds_dev = load_data(TRAIN_DATA_PATH,DEV_SIZE,RANDOM_SEED)
    tokenizer = WordTokenizer(
            args['max_tokens'],
            args['min_freq'],
            args['pattern'],
            args['lower'],
            args['process_accents'],
            unk_token='[UNK]'
        )
    freqs = tokenizer.train(ds_train)
    ds_dev_tokenized, unk_counts = tokenizer.tokenize(ds_dev)
    save_results(freqs,ds_dev_tokenized,unk_counts)



if __name__ == '__main__':
    main()