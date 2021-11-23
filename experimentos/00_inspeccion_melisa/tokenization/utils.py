import numpy as np
import pandas as pd

def get_train_dev_idx(N,dev_size=.2,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    if dev_size == 0:
        return rand_idx

    N_train = int(N * (1-dev_size))
    if N_train == N:
        print('Warning: dev_size too small!')
        N_train = N-1
    
    return rand_idx[:N_train], rand_idx[N_train:]


def train_dev_split(ds,dev_size,random_state):

    N_data = len(ds)
    train_idx, dev_idx = get_train_dev_idx(N_data,dev_size,random_state)
    dev_dataset = ds.iloc[dev_idx].reset_index(drop=True)
    train_dataset = ds.iloc[train_idx].reset_index(drop=True)

    return train_dataset, dev_dataset


def load_data(path,dev_size,random_state):
    print('Loading train data...')
    ds = pd.read_csv(
        path,
        lineterminator='\n',
        sep=',',
        usecols=['review_content'],
        dtype={'review_content': str},
        squeeze=True
    )

    ds_train, ds_dev = train_dev_split(ds,dev_size=dev_size,random_state=random_state)
    
    return ds_train, ds_dev