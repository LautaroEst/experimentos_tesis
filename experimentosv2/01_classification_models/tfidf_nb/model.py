import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class Classifier(object):

    def __init__(self,nclasses,pattern,ngram_range,min_df,max_tokens):
        
        self.nclasses = nclasses
        self.vec = TfidfVectorizer(
                            input='content', encoding='utf-8', 
                            decode_error='strict', strip_accents=None, lowercase=False, 
                            preprocessor=None, tokenizer=None, analyzer='word', 
                            stop_words=None, token_pattern=pattern, 
                            ngram_range=ngram_range, max_df=1.0, min_df=min_df, max_features=max_tokens, 
                            vocabulary=None, binary=False, dtype=float, norm='l2', use_idf=True, 
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
        # Pasamos a minúscula todo
        ds = ds.str.lower()
        # Sacamos todos los acentos
        for rep, rep_with in [('[óòÓöøôõ]','o'), ('[áàÁäåâãÄ]','a'), ('[íìÍïîÏ]','i'), 
                            ('[éèÉëêÈ]','e'), ('[úüÚùûÜ]','u'), ('[ç¢Ç]','c'), 
                            ('[ý¥]','y'),('š','s'),('ß','b'),('\x08','')]:
            ds  = ds.str.replace(rep,rep_with,regex=True)
        return ds