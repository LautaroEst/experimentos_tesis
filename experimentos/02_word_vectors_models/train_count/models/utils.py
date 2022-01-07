import os

import pandas as pd

MELISA_UNSUP_PATH = os.path.join(os.getcwd(),"../../../../other_datasets/melisa_unsup.txt")
MELISA_SUP_PATH = os.path.join(os.getcwd(),"../../../../other_datasets/")

def melisa_unsup_reader():
    with open(MELISA_UNSUP_PATH,"r") as f:
        for line in f.readlines():
            yield line.split(" ")


def melisa_sup_reader(nclasses):
    df = pd.read_csv(
        os.path.join(MELISA_SUP_PATH,"melisa_{}classes.csv".format(nclasses)),
        usecols=['review_title','0']
    )
    
    for _, (review, rate) in df.iterrows():
        yield review.split(" "), int(rate)