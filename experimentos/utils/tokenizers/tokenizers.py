from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
import re


class RegexTokenizer(object):

    def __init__(self,token_pattern):
        self.token_pattern = re.compile(token_pattern)
        self.vocab = None

    def tokenize(self,sent):
        return self.token_pattern.findall(sent)

    def train(self,corpus):
        pass
        



