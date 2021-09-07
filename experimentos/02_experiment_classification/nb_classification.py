import numpy as np

import sys
sys.path.append('/home/lestien/Documents/Trabajos 2021/melisa/experimentos/')
import utils as ut

from utils.tokenizers import RegexTokenizer, NLTKWordTokenizer, NLTKTweetTokenizer
from utils.classifiers import NaiveBayesClassifier
import pickle


DATA_PATH = '/home/lestien/Documents/Trabajos 2021/melisa/datav2/esp/'

def normalize_dataset(df):
    # Pasamos a minúscula todo
    df['review_content'] = df['review_content'].str.lower()
    # Sacamos todos los acentos
    for rep, rep_with in [('[óòÓöøôõ]','o'), ('[áàÁäåâãÄ]','a'), ('[íìÍïîÏ]','i'), 
                          ('[éèÉëêÈ]','e'), ('[úüÚùûÜ]','u'), ('[ç¢Ç]','c'), 
                          ('[ý¥]','y'),('š','s'),('ß','b'),('\x08','')]:
        df['review_content']  = df['review_content'].str.replace(rep,rep_with,regex=True)
    return df

def train_tokenizers(df_train,max_words,freq_cutoff,ngram_range,unk_token):

    tokenizers = {
        'regex': RegexTokenizer(r'(\w+|[\.,!\(\)"\-:\?/%;¡\$\'¿\\]|\d+)',
                                max_words,freq_cutoff,ngram_range,unk_token),
        'nltk': NLTKWordTokenizer('spanish',max_words,freq_cutoff,ngram_range,unk_token),
        'tweet': NLTKTweetTokenizer(max_words,freq_cutoff,ngram_range,unk_token)
    }


    for name, tknzr in tokenizers.items():
        print('{} tokenizer:'.format(name))
        tknzr.train(df_train['review_content'])
        print()
        
    return tokenizers

def main():
    
    df = ut.load_data(DATA_PATH,'train',nclasses=5).loc[:,['review_content','review_rate']]
    
    df = normalize_dataset(df)
    df_train, df_dev = ut.train_dev_split(df,dev_size=0.1,random_state=2376482)

    max_words = [10000]
    freq_cutoff = [1]
    ngram_range = (1,2)
    unk_token = None
    #reweight = ['none','tfidf','ppmi']
    reweight = ['tfidf','ppmi']

    results = {}

    for mw in max_words:
        for fc in freq_cutoff:
            
            tokenizers = {
                'regex': RegexTokenizer(r'(\w+|[\.,!\(\)"\-:\?/%;¡\$\'¿\\]|\d+)',
                                        mw,fc,ngram_range,unk_token),
                'nltk': NLTKWordTokenizer('spanish',mw,fc,ngram_range,unk_token),
                'tweet': NLTKTweetTokenizer(mw,fc,ngram_range,unk_token)
            }
            
            for name,tknzr in tokenizers.items():
                tknzr.train(df_train['review_content'])
                print('Tokenizer:',name)

                ids_train = tknzr.sentences_to_ids(df_train.loc[:,'review_content'])
                labels_train = df_train.loc[:,'review_rate'].values

                ids_dev = tknzr.sentences_to_ids(df_dev.loc[:,'review_content'])
                y = df_dev.loc[:,'review_rate'].values
                
                for rw in reweight:
                    # Fit:
                    clf = NaiveBayesClassifier(alpha=1.0,num_features=mw,reweight=rw)
                    clf.fit(ids_train,labels_train)
                    # Predict:
                    y_pred = clf.predict(ids_dev)
                    acc = sum(y_pred == y) * 100 / len(y_pred)
                    print('Max words: {}, cutoff frequency: {}, reweight: {}'.format(mw,fc,rw))
                    print('Accuracy: {:.2f}%'.format(acc))
                    print()

                    results['{}_{}'.format(name,rw)] = {
                        'max_words': mw, 'frequency_cutoff':fc,
                        'ngram_range': ngram_range, 'unk_token': unk_token, 'accuracy':acc
                    }
    
            with open('./nb_classification_5_classes_mw{}_fc{}.pkl'.format(mw,fc),'wb') as f:
                pickle.dump(results,f)
            
            
if __name__ == '__main__':
    main()