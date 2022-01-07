from collections import defaultdict
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

MELISA_DATA_PATH = os.path.join(os.getcwd(),"../../datav2/esp/train.csv")
MELISA_UNSUP_PATH = os.path.join(os.getcwd(),"../../other_datasets/melisa_unsup.txt")
MELISA_SUP_PATH = os.path.join(os.getcwd(),"../../other_datasets/")

accents = [
        ('[óòöøôõ]','ó'), ('[áàäåâã]','á'), ('[íìïî]','í'), ('[éèëê]','é'), ('[úùû]','ú'), ('[ç¢]','c'), 
        ('[ÓÒÖØÔÕ]','Ó'), ('[ÁÀÄÅÂÃ]','Á'), ('[ÍÌÏÎ]','Í'), ('[ÉÈËÊ]','É'), ('[ÚÙÛ]','Ù'), ('Ç','C'),
        ('[ý¥]','y'), ('š','s'), ('ß','b'), ('\x08','')
    ]


def read_data():
    df = pd.read_csv(
        MELISA_DATA_PATH,
        lineterminator='\n',
        sep=',',
        usecols=['review_content','review_title'],
        dtype={'review_content': str, 'review_content': str})
    ds = df['review_title'].str.cat(df['review_content'],sep=" ")
    return ds

def tokenize(ds: pd.Series):
    print("Replacing rare characters...")
    for rep, rep_with in accents:
        ds = ds.str.replace(rep,rep_with,regex=True)
    print("Converting to lowecase...")
    ds = ds.str.lower()
    print("Removing non alphanumeric chars...")
    ds = ds.str.replace(r"[^\w]+"," ",regex=True)
    print("Replacing digits...")
    ds = ds.str.replace(r"\d+","DIGITO",regex=True)
    print("Removing multiple whitespaces...")
    ds = ds.str.replace(r"\s+"," ",regex=True)
    print("Splitting...")
    ds = ds.str.split(" ")
    return ds

def get_vocab(ds):
    counts = defaultdict(lambda: 0)
    for review in ds:
        for tk in review:
            counts[tk] += 1
    ds_counts = pd.Series(counts.values(),index=counts.keys())
    sorted_counts = ds_counts.sort_values(ascending=False)
    return sorted_counts

def unsup_process():
    ds = read_data()
    ds = tokenize(ds)
    for review in ds.head(20):
        print(" ".join(review))
        print()
    vocab = get_vocab(ds)
    print(len(vocab))
    with open(MELISA_UNSUP_PATH,"w") as f:
        for review in tqdm(ds):
            f.write(" ".join(review))
            f.write("\n")


def read_sup_data(nclasses):
    df = pd.read_csv(
        MELISA_DATA_PATH,
        lineterminator='\n',
        sep=',',
        usecols=['review_content','review_title','review_rate'],
        dtype={'review_content': str, 'review_title': str, 'review_rate': int})
    ds = df['review_title'].str.cat(df['review_content'],sep=" ")

    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2
        df.loc[df['review_rate'] == 3,['review_rate']] = 1
    else:
        df['review_rate'] = df['review_rate'] - 1
    labels = df['review_rate'].values
    
    return ds, labels


def sup_process():
    
    for n in [2,3,5]:
        ds, labels = read_sup_data(n)
        ds = tokenize(ds)
        df = pd.concat([ds.apply(lambda x: " ".join(x[:-1])),pd.Series(labels)],axis=1)
        df.to_csv(os.path.join(MELISA_SUP_PATH,"melisa_{}classes.csv".format(n)),index=False)

if __name__ == "__main__":
    # unsup_process()
    sup_process()