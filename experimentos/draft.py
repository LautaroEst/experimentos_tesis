import pandas as pd
import numpy as np
import torch
from itertools import tee, islice

from gensim.models.keyedvectors import KeyedVectors

def main():
    ds = pd.Series([
        'Esto es una prueba'.split(' '),
        'para ver si funciona lo que creo'.split(' '),
        'que va a funcionar ahora.'.split(' ')
    ])
    y = np.array([1,2,3])
    # df = pd.concat((ds,pd.Series(y)),keys=['x','y'],axis=1)
    # print(df)
    # df = df.sort_values(by=['x'],key=lambda x: x.str.len(),ascending=False)
    # print(df)
    # sequence_batch = ds.copy()
    # sent_lenghts = sequence_batch.str.len()
    # sorted_idx = sent_lenghts.argsort()[::-1]
    # sorted_sequence_batch = sequence_batch.iloc[sorted_idx].reset_index(drop=True)
    # sorted_sent_lenghts = sent_lenghts.iloc[sorted_idx].tolist()
    # y_pred = np.array([2,1,3])
    # print(sequence_batch)
    # print(sorted_sequence_batch)
    # print(sorted_sent_lenghts)
    # resorted_idx = sorted_idx.argsort()
    # y_pred = y_pred[resorted_idx]
    # print(y_pred)

    vocab = {
        '<pad>': 0, 
        '<unk>':1,
        'elle': 2,
        'lalo': 3,
        'lolo': 4,
        'mimito': 5,
        'me': 6
    }
    model = torch.nn.Sequential(torch.nn.Embedding(len(vocab),300,padding_idx=0),
            torch.nn.Linear(4,2))

    wordvectors_file_vec = '/home/lestien/Documents/Trabajos 2021/melisa/word_embeddings_models/fasttext-sbwc.vec'
    cantidad = 10
    
    idx2tk = {idx:tk for tk, idx in vocab.items()}
    idx2tk.pop(0)
    idx2tk.pop(1)
    wordvectors = KeyedVectors.load_word2vec_format(wordvectors_file_vec, limit=cantidad)

    def window_gen(word,min_len,max_len):
        return (word[i-n:i] for n in range(min_len,max_len+1) for i in range(n,len(word)+1))

    print(dict(model[0].named_parameters()))

    with torch.no_grad():
        for idx, tk in idx2tk.items():
            try:
                model[0].weight[idx,:] = torch.from_numpy(wordvectors[tk].copy()).float()
                print(tk,'está en el vocab')
            except KeyError:
                v = np.zeros(embedding_dim,dtype=float)
                print(tk,'no está en el vocab')
                for w in window_gen(tk,3,6):
                    try:
                        v += wordvectors[w].copy()
                        print(w,'está en el vocab')
                    except KeyError:
                        v += np.random.randn(embedding_dim)
                        print(w,'no está en el vocab')
                model[0].weight[idx,:] = torch.from_numpy(v).float()

    print(dict(model[0].named_parameters()))
    print(wordvectors.key_to_index.keys())




import argparse
import re

parser = argparse.ArgumentParser()
#parser.add_argument('--pattern', type=str, required=True)
parser.add_argument('--cased', action='store_true', required=False, default=False)


if __name__ == '__main__':
    #main()
    args = parser.parse_args()
    # pattern = """(\\w+|[\\.,!\\(\\)"\\-:\\?/%;¡\\$'¿\\\\]|\\d+)"""
    # pattern = r"(\w+|[\.,!\(\)\"\-:\?/%;¡\$'¿\\]|\d+)"
    # #re.compile(pattern)
    # re.compile(args.pattern)
    # print(pattern == args.pattern)
    # print(pattern)
    # print(args.pattern)
    print(args.cased)
