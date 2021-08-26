import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix


def pca_viz(X,vocab,words):
    """
    Visualización con PCA de las palabras words, cuya representación
    se encuentran en las filas de X.
    X: csr_matrix de Nw filas y Nf columnas (número de palabras por
    número de features)
    """
    idx = [vocab[w] for w in words]
    if isinstance(X,csr_matrix):
        X = X[idx,:].toarray()
    else:
        X = X[idx,:]
    X = X - X.mean(axis=0,keepdims=True)
    U, S, VT = np.linalg.svd(X,full_matrices=False)
    X_r = U[:,:2] * S[:2]
    
    fig, ax = plt.subplots(1,1,figsize=(9,9))
    ax.scatter(X_r[:,0],X_r[:,1])
    
    for i, w in enumerate(words):
        ax.text(X_r[i,0], X_r[i,1], w)
        
    return fig, ax
