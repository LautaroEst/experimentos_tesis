from functools import total_ordering
import os
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import torch
from torch import nn
from elmoformanylangs import Embedder
from tqdm import tqdm


class ELMOEmbedding(nn.Module):

    def __init__(self,tokenizer,embeddings_path,elmo_batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_dim = 1024
        self.num_embeddings = "?"
        self.pad_idx = tokenizer.vocab[tokenizer.pad_token]
        self.embedder = Embedder(embeddings_path,elmo_batch_size)

    def forward(self,batch_sents):
        batch_sents, attention_mask = self.tokenizer(batch_sents)
        embeddings = self.embedder.sents2elmo(batch_sents)
        max_len = max([e.shape[0] for e in embeddings])
        batch_size = len(batch_sents)
        embedding_dim = self.embedding_dim
        embeddings_batch = torch.full((batch_size,max_len,embedding_dim),self.pad_idx,dtype=torch.float)
        for i, e in enumerate(embeddings):
            embeddings_batch[i,:e.shape[0],:] = torch.from_numpy(e)
        embeddings_batch = embeddings_batch.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return embeddings_batch, attention_mask


class FastTextEmbedding(nn.Module):

    def __init__(self,embeddings,tokenizer,embeddings_path):
        super().__init__()
        filename, self.embedding_dim, total_embeddings, min_subword, max_subword = get_dim_num_path(embeddings)
        embeddings_path = os.path.join(embeddings_path,filename)
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.vocab[tokenizer.pad_token]
        self.num_embeddings = len(tokenizer.vocab)
        emb = nn.Embedding(self.num_embeddings,self.embedding_dim,padding_idx=self.pad_idx)

        emb, self.found_prop = self.load_embeddings(embeddings_path,emb,tokenizer.vocab,
                            total_embeddings,min_subword,max_subword)
        
        self.emb = emb


    def load_embeddings(self,embeddings_path,emb_layer,vocab,total_embeddings,min_subword,max_subword):
        idx2tk = {idx:tk for tk, idx in vocab.items()}
        idx2tk.pop(0)
        idx2tk.pop(1)

        embedding_dim = self.embedding_dim
        print("Loading {} embeddings...".format(total_embeddings))
        wordvectors = KeyedVectors.load_word2vec_format(embeddings_path, limit=total_embeddings)


        def window_gen(word,min_len,max_len):
            return (word[i-n:i] for n in range(min_len,max_len+1) for i in range(n,len(word)+1))

        found_all = 0
        found_some = 0
        with torch.no_grad():
            for idx, tk in tqdm(idx2tk.items()):
                try:
                    emb_layer.weight[idx,:] = torch.from_numpy(wordvectors[tk].copy()).float()
                    found_all += 1
                except KeyError:
                    v = np.zeros(embedding_dim,dtype=float)
                    found_some += 1
                    for w in window_gen(tk,min_subword,max_subword):
                        try:
                            v += wordvectors[w].copy()
                        except KeyError:
                            #v += np.random.randn(embedding_dim)
                            pass
                    if v.sum() == 0:
                        v = np.random.randn(embedding_dim)*0.01
                        found_some -= 1
                    emb_layer.weight[idx,:] = torch.from_numpy(v).float()
        
        found_prop = (found_all+found_some) / len(idx2tk) * 100
        print("Found {} words and {} subwords over {} (~{}%)".format(
            found_all,found_some,len(idx2tk),int(found_prop)))
        return emb_layer, found_prop


    def forward(self,batch_sents):
        batch_ids, attention_mask = self.tokenizer(batch_sents)
        device = self.emb.weight.device
        batch_ids = batch_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        embeddedings = self.emb(batch_ids)
        return embeddedings, attention_mask



class WordEmbedding(nn.Module):

    def __init__(self,embeddings,tokenizer,embeddings_path,embedding_dim=None):
        super().__init__()
        if embeddings is None:
            self.embedding_dim = embedding_dim
            load_from_file = False
        else:
            filename, self.embedding_dim, total_embeddings = get_dim_num_path(embeddings)
            embeddings_path = os.path.join(embeddings_path,filename)
            load_from_file = True
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.vocab[tokenizer.pad_token]
        self.num_embeddings = len(tokenizer.vocab)
        emb = nn.Embedding(self.num_embeddings,self.embedding_dim,padding_idx=self.pad_idx)

        if load_from_file:
            emb, self.found_prop = self.load_embeddings(embeddings_path,tokenizer.vocab,total_embeddings,emb)
        
        self.emb = emb

    def load_embeddings(self,embeddings_path,vocab,total_embeddings,emb):
        idx2tk = {idx:tk for tk, idx in vocab.items()}
        idx2tk.pop(0)
        idx2tk.pop(1)

        embedding_dim = self.embedding_dim
        print("Loading {} embeddings...".format(total_embeddings))
        wordvectors = KeyedVectors.load_word2vec_format(embeddings_path, limit=total_embeddings)
        embeddings_found = 0
        with torch.no_grad():
            for idx, tk in tqdm(idx2tk.items()):
                try:
                    emb.weight[idx,:] = torch.from_numpy(wordvectors[tk].copy()).float()
                    embeddings_found += 1
                except KeyError:
                    emb.weight[idx,:] = torch.randn(embedding_dim)*0.01
        
        found_prop = embeddings_found/len(idx2tk)*100
        print("Found {}/{} (~{}%) embeddings".format(embeddings_found,len(idx2tk),int(found_prop)))
        return emb, found_prop

    def forward(self,batch_sents):
        batch_ids, attention_mask = self.tokenizer(batch_sents)
        device = self.emb.weight.device
        batch_ids = batch_ids.to(device=device)
        attention_mask = attention_mask.to(device=device)
        embeddedings = self.emb(batch_ids)
        return embeddedings, attention_mask


def get_dim_num_path(name):
    name2dim = {
        "word2vec300": ('SBW-vectors-300-min5.txt', 300, 1000653),
        "glove300": ('glove-sbwc.i25.vec', 300, 855380),
        "fasttext300": ('fasttext-sbwc.vec', 300, 855380, 3, 6)
    }
    return name2dim[name]