import numpy as np
import pandas as pd
from utils.evaluation import k_fold_validation
import re
from utils import Melisa2Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class SoftMaxClassifier(object):
    """
    Implementación de un clasificador con word-by-category embeddings + CBOW + Softmax
    """

    def __init__(self,tokenizer=None,vocab=None,max_features=None,ngram_range=(1,1)):
        self.vec = CountVectorizer(tokenizer=tokenizer,vocabulary=vocab,
                                   max_features=max_features,ngram_range=ngram_range,
                                   lowercase=True)
        self.clf = LogisticRegression()

    def train(self,ds_train,y_train):
        X = self.vec.fit_transform(ds_train)
        y_train = y_train.astype(int)
        y_one_hot = np.zeros((y_train.size,y_train.max()+1),dtype=float)
        y_one_hot[np.arange(y_train.size),y_train] = 1.
        self.W = X.minimum(1).T.dot(y_one_hot)
        X_train = X.dot(self.W)
        self.clf.fit(X_train,y_train)

    def predict(self,ds_val):
        X_val = self.vec.transform(ds_val).astype(float)
        X_val = X_val.dot(self.W)
        y_pred = self.clf.predict(X_val)
        return y_pred


def main():

    df_all = Melisa2Dataset().get_train_dataframe(usecols=['review_content','review_rate'])#.sample(n=10000,random_state=random_seed).reset_index(drop=True)
    random_seed = 12738

    ngram_range = (1,3)
    max_features = 100000
    token_pattern = re.compile(r'[\w]+|[!¡¿\?\.,\'"]')
    tokenizer = lambda x: token_pattern.findall(x)
    model = SoftMaxClassifier(tokenizer=tokenizer,
                            max_features=max_features,
                            ngram_range=ngram_range)

    score = k_fold_validation(model,df_all,5,random_state=random_seed,metrics='accuracy')

    print('Accuracy obtenida con ngram_range={}, max_features={}:'.format(ngram_range,max_features))
    print('Accuracy: {:.2f}%'.format(score['accuracy']*100))



if __name__ == '__main__':
    main()