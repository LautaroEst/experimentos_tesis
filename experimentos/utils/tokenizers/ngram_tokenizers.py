from collections import Counter, defaultdict
from itertools import chain, tee, islice
import re
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer

class NgramTokenizerBase(object):

    unk_idx = 0

    def __init__(self,max_words,freq_cutoff,ngram_range,unk_token):
        self.max_words = max_words
        self.freq_cutoff = freq_cutoff
        self.ngram_range = ngram_range
        
        if isinstance(unk_token,str) or unk_token is None:
            self.unk_token = unk_token
        else:
            raise TypeError('unk_token must be string or None.')
      
    @staticmethod
    def get_ngrams(doc,min_ngram,max_ngram):
        for n in range(min_ngram,max_ngram+1):
            tlst = doc
            while True:
                a, b = tee(tlst)
                l = tuple(islice(a, n))
                if len(l) == n:
                    yield ' '.join(l)
                    next(b)
                    tlst = b
                else:
                    break

    def pre_tokenize(self,doc):
        raise NotImplementedError('tokenize method not implemented')


    def tokenize(self,doc):
        vocab = self.vocab
        get_ngrams, pre_tokenize = self.get_ngrams, self.pre_tokenize
        min_ngram, max_ngram = self.ngram_range
        if self.unk_token:
            unk_token = self.unk_token
            tokens = [tk if tk in vocab else unk_token for tk in get_ngrams(pre_tokenize(doc),min_ngram,max_ngram)]
        else:
            tokens = [tk for tk in get_ngrams(pre_tokenize(doc),min_ngram,max_ngram) if tk in vocab]

        return tokens

    
    def train(self,corpus):
        # max_words, freq_cutoff = self.max_words, self.freq_cutoff
        # ngrams_generator = (self.get_ngrams(self.pre_tokenize(sent)) for sent in corpus)
        # word_freq = Counter(chain.from_iterable(ngrams_generator))
        # valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        # if self.unk_token:
        #     top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:max_words-1]
        #     self.vocab = {word: idx for idx, word in enumerate(top_k_words,1)}
        #     self.vocab[self.unk_token] = self.unk_idx
        # else:
        #     top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:max_words]
        #     self.vocab = {word: idx for idx, word in enumerate(top_k_words)}
        # self.idx2tk = {idx:tk for tk,idx in self.vocab.items()}
        # print('Vocab size: {}'.format(len(self.vocab)))

        max_words, freq_cutoff = self.max_words, self.freq_cutoff
        get_ngrams, pre_tokenize = self.get_ngrams, self.pre_tokenize
        min_ngram, max_ngram = self.ngram_range
        word_freq = defaultdict(lambda : 0)
        for sent in corpus:
            for word in get_ngrams(pre_tokenize(sent),min_ngram,max_ngram):
                word_freq[word] += 1
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        if self.unk_token:
            top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:max_words-1]
            self.vocab = {word: idx for idx, word in enumerate(top_k_words,1)}
            self.vocab[self.unk_token] = self.unk_idx
        else:
            top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:max_words]
            self.vocab = {word: idx for idx, word in enumerate(top_k_words)}
        self.idx2tk = {idx:tk for tk,idx in self.vocab.items()}
        print('Vocab size: {}'.format(len(self.vocab)))

    
    def sentences_to_ids(self,corpus):
        vocab = self.vocab
        tokenize = self.tokenize
        ids = [[vocab[tk] for tk in tokenize(doc)] for doc in corpus]
        return ids
    
    def decode_ids(self,ids):
        sentence = [self.idx2tk[idx] for idx in ids]
        return sentence


class RegexTokenizer(NgramTokenizerBase):

    def __init__(self,token_pattern,max_words,freq_cutoff,ngram_range,unk_token):
        self.token_pattern = re.compile(token_pattern)
        super().__init__(max_words,freq_cutoff,ngram_range,unk_token)
      
    def pre_tokenize(self,doc):
        return self.token_pattern.findall(doc)


class NLTKWordTokenizer(NgramTokenizerBase):

    def __init__(self,language,max_words,freq_cutoff,ngram_range,unk_token):
        self.language = language
        super().__init__(max_words,freq_cutoff,ngram_range,unk_token)
      
    def pre_tokenize(self,doc):
        return word_tokenize(doc,self.language)


class NLTKTweetTokenizer(NgramTokenizerBase):

    def __init__(self,max_words,freq_cutoff,ngram_range,unk_token):
        self._tweet_tokenizer = TweetTokenizer(strip_handles=True,reduce_len=True)
        super().__init__(max_words,freq_cutoff,ngram_range,unk_token)
      
    def pre_tokenize(self,doc):
        return self._tweet_tokenizer.tokenize(doc)