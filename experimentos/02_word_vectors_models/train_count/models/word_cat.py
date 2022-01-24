import pandas as pd
from utils import *
import re
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from scipy.sparse import coo_matrix
from io import StringIO
import numpy as np


def word_by_cat(reader,freq_cutoff=1,max_words=10000,nclasses=5):
    
    word_counts = defaultdict(lambda: 0) # Diccionario de word counts
    cat_counts_dict = defaultdict(lambda: np.array([0 for _ in range(nclasses)])) # Diccionario de coocurrencias

    # Para cada documento...
    for i, (doc, c) in enumerate(reader(nclasses)):
        # ...me paro en un token...
        for tk in doc:
            word_counts[tk] += 1
            cat_counts_dict[tk][c] += 1
    
    # Ordeno las palabras según la cantidad de veces que aparecieron:
    sorted_words = sorted(
        [word for word, count in word_counts.items() if count >= freq_cutoff],
        key=word_counts.get,
        reverse=True
    )[:max_words]

    # Guardo en un vocabulario las palabras frecuentes
    vocab = {tk:idx for idx, tk in enumerate(sorted_words)}

    # Creo la matriz de coocurrencias con ese vocabulario
    vocab_size = len(vocab)
    X = np.zeros((vocab_size,nclasses),dtype=float)
    for tk, word_vector in cat_counts_dict.items():
        if tk in vocab:
            X[vocab[tk],:] = word_vector

    # Devuelvo la matriz y el vocabulario
    return X, vocab


if __name__ == "__main__":
    sample = pd.read_csv(StringIO("""review_title,0
no fue lo esperado esta de muy baja calidad no fue lo esperado,0.0
malo reconozco que fue muy económico pero su durabilidad fue muy corta,0.0
excelente muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bien muy bi,1.0
malo no me fue útil para mí diseño muy bueno,0.0
no fue lo esperado no fue lo estipulado solo eso voy aclarar,0.0
malo no me fue muy útil para lo que necesitaba,0.0
mal no fue mi tono no tiene buena duración no fue lo que esperaba,0.0
malo no fue el producto deseado,0.0
no muy bueno no muy bueno no muy bueno no muy bueno,0.0"""),sep=",")
    def sample_gen(nclasses):
        for _, (review, rate) in sample.iterrows():
            yield review.split(" "), int(rate)
    X, vocab = word_by_cat(sample_gen,freq_cutoff=2,nclasses=2)
    print(X)
    print(vocab)

