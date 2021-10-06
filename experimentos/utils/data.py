import pandas as pd
import numpy as np

def load_data(DATA_PATH,split='train',nclasses=2):
    
    path = DATA_PATH + split + '.csv'
    df = pd.read_csv(path,
                     lineterminator='\n',
                     sep=',',
                     usecols=['review_content','review_title','review_rate'],
                     dtype={'review_content': str, 'review_title': str, 'review_rate': int})
    
    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1.
        print('Dataset cargado para 2 clases (malo=0, bueno=1)')
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2.
        df.loc[df['review_rate'] == 3,['review_rate']] = 1.
        print('Dataset cargado para 3 clases (malo=0, medio=1, bueno=2)')
    elif nclasses == 5:
        print('Dataset cargado para 5 clases (muy malo=1, malo=2, medio=3, bueno=4 muy bueno=5)')
    else:
        raise TypeError('nclasses must be either 2, 3 or 5')
        
    print('Num samples per category:')
    print(df['review_rate'].value_counts().sort_index())
    return df


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


def train_dev_split(df,dev_size,random_state):

    N_data = len(df)
    train_idx, dev_idx = get_train_dev_idx(N_data,dev_size,random_state)
    dev_dataset = df.iloc[dev_idx,:].reset_index(drop=True)
    train_dataset = df.iloc[train_idx,:].reset_index(drop=True)

    return train_dataset, dev_dataset