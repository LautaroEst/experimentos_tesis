from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class NaiveBayesClassifier(object):

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

    def train(self,ds,y,eval_every=None,dev=None):
        X_train = self.vec.fit_transform(ds)
        y_train = y
        self.clf.fit(X_train,y_train)

    def predict(self,ds):
        X = self.vec.transform(ds)
        y_predict = self.clf.predict(X)
        return y_predict
