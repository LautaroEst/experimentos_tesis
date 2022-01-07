import os
import pandas as pd
from tqdm import tqdm
import re

MELISA_UNSUP_PATH = os.path.join(os.getcwd(),"../../../other_datasets/melisa_unsup.txt")
MELISA_SUP_PATH = os.path.join(os.getcwd(),"../../../other_datasets/")
SBWC_PATH = os.path.join(os.getcwd(),"../../../other_datasets/spanish_billion_words/")


def melisa_unsup_reader():
    with open(MELISA_UNSUP_PATH,"r") as f:
        lines = f.readlines()
        for line in tqdm(lines,total=len(lines)):
            yield line.split(" ")[:-1]

def melisa_sup_reader(nclasses):
    df = pd.read_csv(
        os.path.join(MELISA_SUP_PATH,"melisa_{}classes.csv".format(nclasses)),
        usecols=['review_title','0'],
        dtype={'review_title': str, '0': float}
    )
    df = df.dropna().reset_index(drop=True)
    
    for _, (review, rate) in tqdm(df.iterrows(),total=len(df)):
        review = review.split(" ")
        rate = int(rate)
        yield review, rate
            

def sbwc_reader():
    filenames = sorted(os.listdir(SBWC_PATH))
    pattern = re.compile(r"[^\s]+")
    for filename in tqdm(filenames):
        with open(os.path.join(SBWC_PATH,filename),"r") as f:
            for line in f.readlines():
                line = pattern.findall(line.lower())
                yield line

def melisa_sbwc_reader():
    for line in sbwc_reader():
        yield line
    for line in melisa_unsup_reader():
        yield line
    
