import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


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

def tfidf_old(X):
    df = np.bincount(X.indices, minlength=X.shape[1])/ X.shape[0]
    n = X.shape[1]
    idf = np.zeros_like(df)
    idf[df != 0] = np.abs(np.log(df[df != 0]))
    tf = sparse_matrix_by_column(X,np.asarray(X.sum(axis=1)).squeeze(),op=np.divide)
    X_tfidf = sparse_matrix_by_row(tf,idf,op=np.multiply)
    return X_tfidf


def tfidf_reweight(X):
    n = X.shape[1]

    df = np.diff(X.indptr) / n
    idf = np.zeros_like(df)
    idf[df != 0] = np.abs(np.log(df[df != 0]))
    
    # # Opción alternativa (sklearn):
    # df = np.diff(X.indptr)
    # idf = np.log((1+n) / (1+df)) + 1
    
    tf = sparse_matrix_by_row(X,np.asarray(X.sum(axis=0)).squeeze(),op=np.divide)
    X_tfidf = sparse_matrix_by_column(tf,idf,op=np.multiply)
    return X_tfidf


def ppmi_reweight(X):
    sum_all = X.sum()
    count_col = np.asarray(X.sum(axis=1)).squeeze()
    count_row = np.asarray(X.sum(axis=0)).squeeze()
    oe = sparse_matrix_by_column(sparse_matrix_by_row(X,count_row,op=np.divide),count_col,op=np.divide) * sum_all
    pmi = sparse_matrix_log(oe)
    ppmi = pmi.maximum(0)
    return ppmi


if __name__ == "__main__":
    X = np.array([
        [10, 10, 10, 10],
        [10, 10, 10,  0],
        [10, 10,  0,  0],
        [ 0,  0,  0,  1]
    ])
    X_csr = csr_matrix(X)
    X_tfidf = tfidf_reweight(X_csr)
    X_ppmi = ppmi_reweight(X_csr)
    print(X_ppmi.toarray())
    # X1 = sparse_matrix_by_row(X_csr,np.asarray(X_csr.sum(axis=0)).squeeze(),op=np.divide)
    # print(X1.toarray())