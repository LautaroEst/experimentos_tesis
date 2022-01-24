from utils import *
import re
from collections import defaultdict
from itertools import chain
from tqdm import tqdm
from scipy.sparse import coo_matrix

# %%
def word_by_word(reader,w=2,freq_cutoff=1,max_words=10000):
    
    word_counts = defaultdict(lambda: 0) # Diccionario de word counts
    coocourrence_dict = defaultdict(lambda: 0) # Diccionario de coocurrencias

    # Para cada documento...
    for doc in reader():
        # ...me paro en un token...
        for i, tk in enumerate(doc):
            word_counts[tk] += 1
            # ...y veo el contexto:
            for context_tk in chain(doc[max(0,i-w):i],doc[i+1:i+w+1]):
                coocourrence_dict[(tk,context_tk)] += 1
    
    # Ordeno las palabras según la cantidad de veces que aparecieron:
    sorted_words = sorted(
        [word for word, count in word_counts.items() if count >= freq_cutoff],
        key=word_counts.get,
        reverse=True
    )[:max_words]

    # Guardo en un vocabulario las palabras frecuentes
    vocab = {tk:idx for idx, tk in enumerate(sorted_words)}

    # Creo la matriz de coocurrencias con ese vocabulario
    i, j, data = [], [], []
    for (tk, context_tk), count in coocourrence_dict.items():
        if tk in vocab and context_tk in vocab:
            i.append(vocab[tk])
            j.append(vocab[context_tk])
            data.append(count)

    vocab_size = len(vocab)

    # Devuelvo la matriz y el vocabulario
    return coo_matrix((data, (i,j)),shape=(vocab_size,vocab_size)).tocsr(), vocab


# %%


if __name__ == "__main__":
    sample = """A continuación escucharemos una canción típica del folclore argentino, recopilada por un gran investigador y antropólogo: el licenciado Gustavo Pérez y Alonso. 
    Como buen científico, Pérez y Alonso cultivaba la duda filosófica, se cuestionaba todo constantemente, cultivaba la duda. 
    Sin ir más lejos, firmaba sus libros en vez de "Pérez y Alonso", "Pérez o Alonso"."""
    sample = re.sub(r"[^\w\n]+"," ",sample.lower())
    sample = re.sub(r"(\s*)(\n+)(\s*)","\n",sample)
    sample = [re.split(r"\s+",doc) for doc in re.split(r"\n",sample)]
    def reader():
        for doc in sample:
            yield doc
    mat, vocab = word_by_word(reader,w=2,freq_cutoff=2)
    print(mat.toarray())
    print(vocab)