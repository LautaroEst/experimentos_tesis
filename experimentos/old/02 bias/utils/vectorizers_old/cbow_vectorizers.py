from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from .reweighting import _get_reweight_fn


def _filter_by_frequency(X, full_vocab, reshape_X_fn, get_freqs_fn, 
min_count=0., max_count=None, max_words=None):
    """ Función privada para limitar tokens por frecuencia. Modifica
    tanto el vocabulario como la matriz de coocurrencias que se le pasa. """

    freqs = get_freqs_fn(X)
    sorted_indices = np.argsort(freqs)[::-1]
    sorted_frequencies = freqs[sorted_indices]

    if min_count <= 0 and max_count is None:
        mask = np.ones(X.shape[0],dtype=bool)

    else:
        if max_count is None:
            max_count = np.inf
        mask = np.logical_and(sorted_frequencies <= max_count, 
                                sorted_frequencies >= min_count)

    sorted_indices = sorted_indices[mask]
    
    if max_words is not None:
        sorted_indices = sorted_indices[:max_words]
    
    X = reshape_X_fn(X,sorted_indices)
    idx_to_tk = {idx:tk for tk,idx in full_vocab.items()}
    vocab = {idx_to_tk[idx]:i for i,idx in enumerate(sorted_indices)}

    return X, vocab


def _check_tokenizer(tokenizer):
    """ Función privada para checkear el tokenizer """

    if tokenizer is None:
        tokenizer = lambda x: x
    elif not callable(tokenizer):
        raise ValueError('Tokenizer must be callable or None.')

    return tokenizer


def word_by_category_cooccurrence(corpus, labels, tokenizer=None,
    min_count=0., max_count=None, max_words=None):
    """
    Devuelve la matriz de coocurrencias entre palabras y la categoría a la que pertence 
    el documento. Es decir, las filas de la matriz son las palabras y las columnas son
    todas las categorías posibles, y todas las entradas de la matriz contienen la cuenta
    de cuántas veces apareció la palabra en un documento de cada categoría.
    """

    # Definiciones de para el vocabulario y el diccionario de coocurrencias:
    categories = sorted(set(labels)) # Se asume que los labels son 0, 1, ..., len(categories)
    cooccurrences_dict = defaultdict(float)
    full_vocab = defaultdict()
    full_vocab.default_factory = full_vocab.__len__

    # Se checkea si tokenizer es válido:
    tokenizer = _check_tokenizer(tokenizer)

    # Cuento coocurrencias con las etiquetas:
    for doc, label in zip(tqdm(corpus), labels):
        for tk in tokenizer(doc):
            cooccurrences_dict[(full_vocab[tk],label)] += 1.

    full_vocab = dict(full_vocab)
    i, j = zip(*cooccurrences_dict.keys())
    data = list(cooccurrences_dict.values())
    X = coo_matrix((data, (i,j)),shape=(len(full_vocab),len(categories)))

    # Limito por frecuencia o por tope máximo de palabras
    def get_freqs(X):
        return X.sum(axis=1).A1.reshape(-1)

    def reshape_X(X,mask_or_indices):
        X = X[mask_or_indices,:]
        return X 

    # Limito por frecuencia:
    X, vocab = _filter_by_frequency(X.tocsr(), full_vocab, reshape_X, get_freqs, 
                                min_count, max_count, max_words)
    
    return X, vocab


def _append_unk(X,vocab,unk_token):

	if unk_token not in vocab.keys():
		X.resize(X.shape[0]+1,X.shape[1])
		unk_idx = X.shape[0]-1
	else:
		unk_idx = vocab[unk_token]
	
	return X, unk_idx


class WordByCategoryCBOWVectorizer(object):

    def __init__(self,*args,**kwargs):
        self.sklearn_vec = CountVectorizer(*args,**kwargs)


    def fit_transform(self,ds_train,y_train):
        X = self.sklearn_vec.fit_transform(ds_train)
        y_train = y_train.astype(int)
        y_one_hot = np.zeros((y_train.size,y_train.max()+1),dtype=float)
        y_one_hot[np.arange(y_train.size),y_train] = 1.
        self.W = X.minimum(1).T.dot(y_one_hot)
        X_train = X.dot(self.W)
        return X_train

    def transform(self,ds_test):
        X = self.sklearn_vec.transform(ds_test)
        X_test = X.dot(self.W)
        return X_test

    def fit(self,ds_train,y_train):
        _ = self.fit_transform(ds_train,y_train)








""" 
class WordByCategoryCBOWVectorizer(object):

    def __init__(self, tokenizer=None, min_count=0., max_count=None, 
                max_features=None, unk_token=None, reweighting=None):
        self.tokenizer = tokenizer
        self.min_count = min_count
        self.max_count = max_count
        self.max_features = max_features
        self.unk_token = unk_token
        self.reweighting = _get_reweight_fn(reweighting)
        
    def train(self,corpus,labels):
        X, vocab = word_by_category_cooccurrence(corpus,labels,self.tokenizer,
                                self.min_count,self.max_count,self.max_features)
        X, unk_idx = _append_unk(X,vocab,self.unk_token)
        X = self.reweighting(X.toarray())
        self.X = X
        self.vocab = vocab
        self.unk_idx = unk_idx

    def vectorize_doc(self,doc):
        vocab_get = self.vocab.get
        unk_idx = self.unk_idx
        X = self.X
        indices = [vocab_get(tk,unk_idx) for tk in self.tokenizer(doc)]
        cbow_vec = X[indices,:].sum(axis=0)
        return cbow_vec

    def vectorize_corpus(self,corpus):
        cbow_mat = np.zeros((len(corpus),self.X.shape[1]))
        for i,doc in enumerate(tqdm(corpus)):
            cbow_mat[i,:] = self.vectorize_doc(doc)
        return cbow_mat """
