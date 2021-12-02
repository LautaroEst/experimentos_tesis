from gensim.models.keyedvectors import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import os
EMBEDDINGS_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/pretrained_models/'
embeddings_file_paths = {
    'fasttext': EMBEDDINGS_PATH + 'fasttext-sbwc.vec',
    'glove': EMBEDDINGS_PATH + 'glove-sbwc.i25.vec',
    'word2vec': EMBEDDINGS_PATH + 'SBW-vectors-300-min5.txt'
}

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



def load_fasttext(emb_layer,idx2tk,wordvectors,embedding_dim,min_subword,max_subword):

    def window_gen(word,min_len,max_len):
        return (word[i-n:i] for n in range(min_len,max_len+1) for i in range(n,len(word)+1))

    found_all = 0
    found_some = 0
    with torch.no_grad():
        for idx, tk in tqdm(idx2tk.items()):
            try:
                emb_layer.weight[idx,:] = torch.from_numpy(wordvectors[tk].copy()).float()
                found_all += 1
            except KeyError:
                v = np.zeros(embedding_dim,dtype=float)
                found_some += 1
                for w in window_gen(tk,min_subword,max_subword):
                    try:
                        v += wordvectors[w].copy()
                    except KeyError:
                        #v += np.random.randn(embedding_dim)
                        pass
                if v.sum() == 0:
                    v = np.random.randn(embedding_dim)
                    found_some -= 1
                emb_layer.weight[idx,:] = torch.from_numpy(v).float()
    
    print("Found {} words and {} subwords".format(found_all,found_some))
    return emb_layer


def load_glove_word2vec(embedding_layer,idx2tk,wordvectors,embedding_dim):
    
    embeddings_found = 0
    with torch.no_grad():
        for idx, tk in tqdm(idx2tk.items()):
            try:
                embedding_layer.weight[idx,:] = torch.from_numpy(wordvectors[tk].copy()).float()
                embeddings_found += 1
            except KeyError:
                embedding_layer.weight[idx,:] = torch.randn(embedding_dim) 
    
    print("Found {}/{} ({:.0}%) embeddings".format(embeddings_found,len(idx2tk),embeddings_found/len(idx2tk)*100))
    return embedding_layer
        

def init_embeddings(model,vocab,embeddings):
    
    idx2tk = {idx:tk for tk, idx in vocab.items()}
    idx2tk.pop(0)
    idx2tk.pop(1)

    embeddings, finetune = embeddings.split('-')
    wordvectors_file_vec = embeddings_file_paths[embeddings]

    embedding_dim = 300
    if embeddings == 'fasttext':
        cantidad = 855380
        print('Loading {} pretrained word embeddings...'.format(cantidad))
        wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
        min_subword = 3
        max_subword = 6
        model.emb = load_fasttext(model.emb,idx2tk,wordvectors,embedding_dim,min_subword,max_subword)
    elif embeddings == 'glove':
        cantidad = 855380
        print('Loading {} pretrained word embeddings...'.format(cantidad))
        wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
        model.emb = load_glove_word2vec(model.emb,idx2tk,wordvectors,embedding_dim)
    elif embeddings == 'word2vec':
        cantidad = 1000653
        print('Loading {} pretrained word embeddings...'.format(cantidad))
        wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)
        model.emb = load_glove_word2vec(model.emb,idx2tk,wordvectors,embedding_dim)

    return model