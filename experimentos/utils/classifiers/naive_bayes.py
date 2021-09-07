from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from scipy.sparse import csr_matrix
from ..vsm import sparse_tfidf, sparse_ppmi
import numpy as np


class NaiveBayesClassifier(object):

    def __init__(self,alpha,num_features,reweight='none'):
        self.clf = MultinomialNB(alpha=alpha)
        self.num_features = num_features
        if reweight in ('none', 'tfidf', 'ppmi'):
            self.reweight = reweight
        else:
            raise ValueError('reweight must be tfidif, ppmi or none')

    def _vectorize(self,tokens_indices):
        indptr = [0]
        indices = []
        data = []
        for doc in tokens_indices:
            for ind, counts in Counter(doc).items():
                indices.append(ind)
                data.append(counts)
            indptr.append(len(indices))
                
        data = np.array(data,dtype=float)
        indices = np.array(indices,dtype=int)
        indptr = np.array(indptr,dtype=int)
        X = csr_matrix((data, indices, indptr), shape=(len(tokens_indices),self.num_features), dtype=float)

        if self.reweight == 'tfidf':
            X = sparse_tfidf(X)
        elif self.reweight == 'ppmi':
            X = sparse_ppmi(X)

        return X

    def fit(self,tokens_indices,labels):
        X = self._vectorize(tokens_indices)
        self.clf.fit(X,labels)

    def predict(self,tokens_indices):
        X = self._vectorize(tokens_indices)
        predicted_labels = self.clf.predict(X)
        return predicted_labels
