import json
import os
import pandas as pd
from gensim.models import KeyedVectors
import numpy as np

files = [
    "capital-common-countries.csv",
    "capital-world.csv",
    "city-in-state.csv",
    "currency.csv",
    "family.csv",
    "gram1-adjective-to-adverb.csv",
    "gram2-opposite.csv",
    "gram5-present-participle.csv",
    "gram6-nationality-adjective.csv",
    "gram7-past-tense.csv",
    "gram8-plural.csv",
    "gram9-plural-verbs.csv"
]

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


def load_vectors(embeddings_file):
    model = KeyedVectors.load_word2vec_format(embeddings_file)
    return model

def setup_matrix(model,vocab):
    
    dim = model.vector_size
    not_founded_words = []
    word_vectors = []
    founded_count = 0
    idx2word = {}
    vocab_size = len(vocab)
    X = np.random.randn(vocab_size,dim) * 0.001
    for word, idx in vocab.items():
        try:
            X[idx,:] = model[word]
            idx2word[founded_count] = word
            founded_count += 1
        except KeyError:
            not_founded_words.append(word)
    if len(not_founded_words) != 0:
        print("Warning: the following words have not been found:")
        for word in not_founded_words[:-1]:
            print(word,end=", ")
        print(not_founded_words[-1])
    return X

def main():
    
    for name, embeddings_file in name2file.items():

        print("Loading embeddings for {}...".format(name))
        model = load_vectors(embeddings_file)
        accuracies = {}
        for data_file in files:
            print("Loading data...")
            df = pd.read_csv("./data/{}".format(data_file))
            vocab = {tk: idx for idx, tk in enumerate(sorted(np.unique(df)))}

            print("Setting up embedding matrix...")
            X = setup_matrix(model,vocab)

            print("Predicting...")
            predictions = []
            for _, (w1,w2,w3,w4) in df.iterrows():
                idx1, idx2, idx3 = vocab[w1], vocab[w2], vocab[w3]
                x1, x2, x3 = X[idx1,:], X[idx2,:], X[idx3,:]
                x4_hat = (x2 - x1 + x3).reshape(-1,1)
                bests_fit = np.argsort(X.dot(x4_hat).reshape(-1))[::-1]
                i = 0
                while i < 3:
                    if not bests_fit[i] in [idx1, idx2, idx3]:
                        break
                    i += 1
                idx4_pred = bests_fit[i]
                idx4_true = vocab[w4]
                predictions.append(idx4_pred == idx4_true)

            accuracy = sum(predictions) / len(predictions)
            print("Accuracy: {:.2f}%".format(accuracy * 100))
            accuracies[data_file.split(".")[0]] = accuracy

        print(accuracies)
        df = pd.DataFrame.from_dict(accuracies,orient="index")
        df.to_csv("./results/{}.csv".format(name))




if __name__ == "__main__":
    main()
    