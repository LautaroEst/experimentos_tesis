from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt

def load_word_vectors_to_array(embeddings_file,words=None):

    print("Loading embeddings...")
    model = KeyedVectors.load_word2vec_format(embeddings_file)

    if words is None:
        random_index = np.random.randint(0,len(model))
        word_vectors = np.array([model[model.index_to_key(idx)] for idx in random_index])
        idx2word = {i: model.index_to_key(idx) for i, idx in enumerate(random_index)}
    else:
        not_founded_words = []
        word_vectors = []
        founded_count = 0
        idx2word = {}
        for word in words:
            try:
                word_vectors.append(model[word])
                idx2word[founded_count] = word
                founded_count += 1
            except KeyError:
                not_founded_words.append(word)
        word_vectors = np.array(word_vectors)
        if len(not_founded_words) != 0:
            print("Warning: the following words have not been found:")
            for word in not_founded_words[:-1]:
                print(word,end=", ")
            print(not_founded_words[-1])
    
    return word_vectors, idx2word


def svd_2d_visualization(embeddings_file,words=None):

    X, idx2word = load_word_vectors_to_array(embeddings_file,words)

    X = X - X.mean(axis=0,keepdims=True)
    U, S, _ = np.linalg.svd(X,full_matrices=False)
    X_r = U[:,:2] * S[:2]
    
    fig, ax = plt.subplots(1,1,figsize=(9,9))
    ax.scatter(X_r[:,0],X_r[:,1])
    
    for i, w in idx2word.items():
        ax.text(X_r[i,0], X_r[i,1], w)
        
    return fig, ax
