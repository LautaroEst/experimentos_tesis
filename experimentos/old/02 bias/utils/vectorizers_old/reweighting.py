import numpy as np

def _get_reweight_fn(reweighting):
	if reweighting == 'oe':
		return observed_over_expected
	elif reweighting == 'ppmi':
		return ppmi
	elif reweighting == 'tfidf':
		return tfidf
	elif reweighting is None:
		return lambda x: x
	else:
		raise ValueError('{} reweighting method not implemented.'.format(reweighting))



def observed_over_expected(X):
    if X.sum() == 0:
        return X
    row_sum = np.sum(X,axis=0,keepdims=True)        
    column_sum = np.sum(X,axis=1,keepdims=True)
    expected = row_sum * column_sum / X.sum()
    observed_expected = np.zeros_like(X,dtype=float)
    np.divide(X,expected,out=observed_expected,where=(expected!=0))
    return observed_expected

def ppmi(X):
    if X.sum() == 0:
        return X
    X = observed_over_expected(X)
    positive_pmi = np.zeros_like(X,dtype=float)
    np.log(X,out=positive_pmi,where=(X>1.))
    return positive_pmi

def tfidf(X):
    """ Hay diferentes versiones de este cálculo. Nosotros hacemos el de Jurafsky un poco modificado:
    tf(w,d) = cantidad de veces que la palabra w apareció en el documento d  / cantidad de veces palabras en el documento d
    idf(w) =  log( (cantidad de documentos total + 1 / (cantidad de documentos en donde apareció w + 1) )
    tfidf(w,d) = tf(w,d) x idf(w)
    """
    tf = np.zeros_like(X,dtype=float)
    colsum = X.sum(axis=0,keepdims=True)
    np.divide(X,colsum,out=tf,where=(colsum!=0.))
    N = X.shape[1]
    df = np.minimum(X,1.).sum(axis=1,keepdims=True)
    idf = np.log((1+N)/(1+df))
    tfidf = tf * idf
    return tfidf












""" 



# Reweight methods:

def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = np.zeros(df.shape)
    mask = expected != 0
    #np.putmask(oe, mask, df[mask] / expected[mask])
    #oe[mask].reshape(1,-1) = (df[mask] / expected[mask]).reshape(-1)
    oe[mask] = df[mask] / expected[mask]
    return oe


def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df


def tfidf(df):
    # Inverse document frequencies:
    doccount = float(df.shape[1])
    freqs = df.astype(bool).sum(axis=1)
    idfs = np.log(doccount / freqs)
    idfs[np.isinf(idfs)] = 0.0  # log(0) = 0
    # Term frequencies:
    col_totals = df.sum(axis=0)
    tfs = np.zeros_like(df)
    mask = col_totals!=0 
    tfs[:,mask] = df[:,mask] / col_totals[mask]
    return (tfs.T * idfs).T """