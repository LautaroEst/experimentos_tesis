import os
from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


name2file = {
    key: os.path.join(os.getcwd(),"../../../pretrained_models/",value) for key, value in [
        ("melisa_wbw_w2", "word_by_word_none_w2_fc5_dim300.vec"), 
        ("melisa_wbw_w4", "word_by_word_none_w4_fc5_dim300.vec"),
        ("melisa_wbw_w8", "word_by_word_none_w8_fc5_dim300.vec"),
        ("melisa_wbw_w16", "word_by_word_none_w16_fc5_dim300.vec"),
        ("melisa_wbw_w2_tfidf", "word_by_word_tfidf_w2_fc5_dim300.vec"), 
        ("melisa_wbw_w4_tfidf", "word_by_word_tfidf_w4_fc5_dim300.vec"),
        ("melisa_wbw_w8_tfidf", "word_by_word_tfidf_w8_fc5_dim300.vec"), 
        ("melisa_wbw_w16_tfidf", "word_by_word_tfidf_w16_fc5_dim300.vec"),
        ("melisa_wbw_w2_ppmi", "word_by_word_ppmi_w2_fc5_dim300.vec"), 
        ("melisa_wbw_w4_ppmi", "word_by_word_ppmi_w4_fc5_dim300.vec"),
        ("melisa_wbw_w8_ppmi", "word_by_word_ppmi_w8_fc5_dim300.vec"), 
        ("melisa_wbw_w16_ppmi", "word_by_word_ppmi_w16_fc5_dim300.vec"),
        
        ("melisa_wbc_n2", "word_by_cat_none_fc5_n2.vec"), 
        ("melisa_wbc_n3", "word_by_cat_none_fc5_n3.vec"),
        ("melisa_wbc_n5", "word_by_cat_none_fc5_n5.vec"), 
        ("melisa_wbc_tfidf_n2", "word_by_cat_tfidf_fc5_n2.vec"), 
        ("melisa_wbc_tfidf_n3", "word_by_cat_tfidf_fc5_n3.vec"), 
        ("melisa_wbc_tfidf_n5", "word_by_cat_tfidf_fc5_n5.vec"), 
        ("melisa_wbc_ppmi_n2", "word_by_cat_ppmi_fc5_n2.vec"), 
        ("melisa_wbc_ppmi_n3", "word_by_cat_ppmi_fc5_n3.vec"),
        ("melisa_wbc_ppmi_n5", "word_by_cat_ppmi_fc5_n5.vec"),
        
        ("sbwc_word2vec300", "SBW-vectors-300-min5.txt"),
        ("sbwc_glove300", "glove-sbwc.i25.vec"),
        ("sbwc_fasttext300", "fasttext-sbwc.vec")
    ]
}


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


def make_tensorboard_visualization(embeddings_name,embeddings_file,words=None):

    X, idx2word = load_word_vectors_to_array(embeddings_file,words)
    
    embeddings_log_dir = os.path.join("log_embeddings",embeddings_name)
    if not os.path.exists(embeddings_log_dir):
        os.mkdir(embeddings_log_dir)
    
    writer = SummaryWriter(log_dir=embeddings_log_dir)
    writer.add_embedding(X,metadata=[idx2word[idx] for idx in range(len(idx2word))])
    writer.close()



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
