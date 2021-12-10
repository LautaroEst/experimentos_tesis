from datasets import load_dataset
import pandas as pd
import numpy as np


def load_cine(nclasses,split='train'):
    CINE_RNDM_SEED = 2374812
    rs = np.random.RandomState(CINE_RNDM_SEED)
    dataset = load_dataset("muchocine")['train']
    print(dataset)
    df = pd.DataFrame(dataset).loc[:,['review_body','review_summary','star_rating']]
    df = df.rename(columns={'review_body':'review_content', 'star_rating': 'review_rate', 'review_summary': 'review_title'})
    N = len(df)
    N_train, N_dev = int(N*0.85), int(N*0.05)
    indices = rs.permutation(N)
    df_train = df.iloc[indices[:N_train],:].reset_index(drop=True)
    df_dev = df.iloc[indices[N_train:N_train+N_dev],:].reset_index(drop=True)
    df_test = df.iloc[indices[N_train+N_dev:],:].reset_index(drop=True)
    if split == 'train':
        return df_train
    elif split == 'dev':
        return df_dev
    elif split == 'test':
        return df_test

def main():
    load_cine(nclasses=2)
    # print(dataset)

if __name__ == "__main__":
    main()