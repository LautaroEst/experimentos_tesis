import numpy as np
from scipy.sparse import csr_matrix, issparse

def sparse_matrix_by_column(X,column,op):
    X = X.copy()
    X.data = op(X.data,np.repeat(column,np.diff(X.indptr)))
    return X

def sparse_matrix_by_row(X,row,op):
    X = X.copy()
    X.data = op(X.data,row[X.indices])
    return X

def sparse_matrix_log(X):
    X = X.copy()
    X.data = np.log(X.data,where=X.data != 0.) # log(0) = 0
    return X 

# Ambas funciones reciben la salida de CountVectorizer

def ppmi(X):
    sum_all = X.sum()
    if issparse(X):
        prob_col = np.asarray(X.sum(axis=1)).squeeze() / sum_all
        prob_row = np.asarray(X.sum(axis=0)).squeeze() / sum_all
        prob_all = X / sum_all
        prob_all = sparse_matrix_by_row(prob_all,prob_row,op=np.divide)
        prob_all = sparse_matrix_by_column(prob_all,prob_col,op=np.divide)
        prob_all = sparse_matrix_log(prob_all).maximum(0)
    else:
        prob_col = X.sum(axis=1,keepdims=True) / sum_all
        prob_row = X.sum(axis=0,keepdims=True) / sum_all
        prob_all = X / sum_all
        prob_all = prob_all / prob_row
        prob_all = prob_all / prob_col
        prob_all = np.maximum(np.log(prob_all,where=prob_all!=0),0)
    return prob_all

def tfidf(X):
    if issparse(X):
        idf = np.log(X.shape[0] / np.asarray(X.astype(bool).sum(axis=0)).squeeze())
        tf = sparse_matrix_by_column(X,np.asarray(X.sum(axis=1)).squeeze(),op=np.divide)
        X_tfidf = sparse_matrix_by_row(tf,idf,op=np.multiply)
    else:
        idf = np.log( X.shape[0] / X.astype(bool).sum(axis=0,keepdims=True))
        tf = X / X.sum(axis=1,keepdims=True)
        X_tfidf = tf * idf
    return X_tfidf