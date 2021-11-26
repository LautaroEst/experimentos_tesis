from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict


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


class CatBOWVectorizer(object):

    pattern = r"(\w+|[\.,!\(\)\"\-:\?/%;¡\$'¿\\]|\d+)"

    def __init__(self,nclasses,ngram_range,max_features):
        self.vec = CountVectorizer(token_pattern=self.pattern,
                ngram_range=ngram_range,max_features=max_features)
        self.labels_names = list(range(nclasses))
    
    def _vectorizer_by_cat(self,X,labels):
        n_cats = len(self.labels_names)
        word_vecs = np.zeros((X.shape[1],n_cats),dtype=float)
        for i,n in enumerate(self.labels_names):
                word_vecs[:,i] = X[labels == n,:].sum(axis=0)
        X_cats = X @ word_vecs
        self.learned_words = word_vecs
        return X_cats

    def fit_transform(self,corpus,labels):
        X = self.vec.fit_transform(corpus)
        X_cats = self._vectorizer_by_cat(X,labels)
        self.learned_mean = X_cats.mean(axis=0,keepdims=True)
        X_cats = X_cats - self.learned_mean
        return X_cats
    
    def transform(self,corpus):
        X = self.vec.transform(corpus)
        X_cats = X @ self.learned_words
        X_cats = X_cats - self.learned_mean
        return X_cats


class VocabVectorizer(object):

    pattern = r"(\w+|[\.,!\(\)\"\-:\?/%;¡\$'¿\\]|\d+)"

    def __init__(self,freq_cutoff,max_tokens,max_sent_len,
                pad_token,unk_token):

        self.freq_cutoff = freq_cutoff
        self.max_tokens = max_tokens
        self.max_sent_len = max_sent_len
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = None

    def create_vocabulary(self,corpus):
        word_freq = defaultdict(lambda : 0)
        fc = self.freq_cutoff
        for sent in corpus:
            for word in sent:
                word_freq[word] += 1
        valid_words = [w for w, v in word_freq.items() if v >= fc]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:self.max_tokens-2]
        vocab = {word: idx for idx, word in enumerate(top_k_words,2)}
        vocab[self.pad_token] = 0
        vocab[self.unk_token] = 1
        self.vocab = vocab
        return vocab

    def fit_transform(self,ds):
        corpus = ds.str.findall(self.pattern)
        vocab = self.create_vocabulary(corpus)
        unk_idx = vocab[self.unk_token]
        max_sent_len = self.max_sent_len
        ds = corpus.apply(lambda sent: [vocab.get(tk,unk_idx) for tk in sent[:max_sent_len]])
        return ds
    
    def fit(self,ds):
        corpus = ds.str.findall(self.pattern)
        vocab = self.create_vocabulary(corpus)
        self.vocab = vocab

    def transform(self,ds):
        ds = ds.str.findall(self.pattern)
        vocab = self.vocab
        unk_idx = vocab[self.unk_token]
        ds = ds.apply(lambda sent: [vocab.get(tk,unk_idx) for tk in sent])
        return ds