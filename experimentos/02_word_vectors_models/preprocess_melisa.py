from collections import defaultdict
import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

MELISA_DATA_PATH = os.path.join(os.getcwd(),"../../datav2/esp/train.csv")
MELISA_UNSUP_PATH = os.path.join(os.getcwd(),"../../other_datasets/melisa_unsup.txt")

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
    print("converting to lowecase...")
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

def main():
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


if __name__ == "__main__":
    main()