import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class Classifier(object):

    pattern = r"(\w+|[\.,!\(\)\"\-:\?/%;¡\$'¿\\]|\d+)"

    def __init__(self,nclasses,ngram_range,max_features):
        
        self.nclasses = nclasses
        self.vec = TfidfVectorizer(
                            input='content', encoding='utf-8', 
                            decode_error='strict', strip_accents=None, lowercase=False, 
                            preprocessor=None, tokenizer=None, analyzer='word', 
                            stop_words=None, token_pattern=self.pattern, 
                            ngram_range=ngram_range, max_df=1.0, min_df=1, max_features=max_features, 
                            vocabulary=None, binary=False, norm='l2', use_idf=True, 
                            smooth_idf=True, sublinear_tf=False
                    )
        
        self.clf = MultinomialNB(alpha=1.,fit_prior=True,class_prior=None)

    def train(self,ds,y):
        ds = self.normalize_dataset(ds)
        X_train = self.vec.fit_transform(ds)
        y_train = y
        self.clf.fit(X_train,y_train)

    def predict(self,ds):
        ds = self.normalize_dataset(ds)
        X = self.vec.transform(ds)
        y_predict = self.clf.predict(X)
        return y_predict

    def normalize_dataset(self,ds):

        accents = [
            ('[óòöøôõ]','ó'), ('[áàäåâã]','á'), ('[íìïî]','í'), ('[éèëê]','é'), ('[úùû]','ú'), ('[ç¢]','c'), 
            ('[ÓÒÖØÔÕ]','Ó'), ('[ÁÀÄÅÂÃ]','Á'), ('[ÍÌÏÎ]','Í'), ('[ÉÈËÊ]','É'), ('[ÚÙÛ]','Ù'), ('Ç','C'),
            ('[ý¥]','y'), ('š','s'), ('ß','b'), ('\x08','')
        ]
        for rep, rep_with in accents:
            ds  = ds.str.replace(rep,rep_with,regex=True)

        ds = ds.str.lower()

        return ds