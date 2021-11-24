from datetime import datetime
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

DATA_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/datav2/'
DATA_ES_PATH = DATA_PATH + 'esp/'
DATA_POR_PATH = DATA_PATH + 'por/'


def load_data(lang='es',split='train',nclasses=2):
    
    data_path = DATA_ES_PATH if lang == 'es' else DATA_POR_PATH
    path = data_path + split + '.csv'
    df = pd.read_csv(path,
                     lineterminator='\n',
                     sep=',',
                     usecols=['review_content','review_rate'],
                     dtype={'review_content': str, 'review_rate': int})
    
    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1.
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2.
        df.loc[df['review_rate'] == 3,['review_rate']] = 1.
    else:
        df['review_rate'] = df['review_rate'] - 1

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


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Argumentos para el entrenamiento:
    parser.add_argument('--nclasses', type=int, required=True)
    parser.add_argument('--lang', type=str, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--devsize', type=float)
    group.add_argument('--test', action='store_true')

    # Argumentos para el tokenizer:
    parser.add_argument('--ngram_range', type=int, nargs=2, required=True)
    parser.add_argument('--max_features', type=int, required=True)
    parser.add_argument('--hidden_size', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--eval_every', type=int, required=True)

    args = vars(parser.parse_args())
    return args


def show_results(
        y_train_predict,
        y_train_true,
        y_dev_predict,
        y_dev_true,
        history,
        nclasses,
        description,
        is_dev
    ):
    
    now = datetime.now()
    title = now.strftime("%Y-%m-%d-%H-%M-%S")
    list_of_labels = list(range(nclasses))
    results_dir = 'results_{}classes'.format(nclasses)

    # Classification Report:
    name = 'Dev' if is_dev else 'Test'
    report = """
{}

Classification Report (Train):
------------------------------
    
{}


Classification Report ({}):
------------------------------
    
{}

    """.format(
        description,
        classification_report(y_train_true,y_train_predict,labels=list_of_labels),
        name,
        classification_report(y_dev_true,y_dev_predict,labels=list_of_labels)
    )

    with open('{}/{}_classification_report.log'.format(results_dir,title),'w') as f:
        f.write(report)

    # Confusion Matrix:
    cm_train = confusion_matrix(y_train_true,y_train_predict,labels=list_of_labels)
    cm_dev = confusion_matrix(y_dev_true,y_dev_predict,labels=list_of_labels)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
    im = ax1.imshow(cm_train,cmap='cividis')
    ax1.set_title('Train Confusion Matrix',fontsize='xx-large')
    im = ax2.imshow(cm_dev,cmap='cividis')
    ax2.set_title('{} Confusion Matrix'.format(name),fontsize='xx-large')

    for ax, cm in [(ax1, cm_train), (ax2, cm_dev)]:
        ticks = list_of_labels
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks,fontsize='xx-large')
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks,fontsize='xx-large')
        
        for i in range(nclasses):
            for j in range(nclasses):
                text = ax.text(j, i, cm[i, j],
                            ha="center", va="center", color="red")
    
    fig.tight_layout()
    plt.savefig('{}/{}_confusion_matrix.png'.format(results_dir,title))

    # Loss and accuracy history:
    with open("{}/{}_history.pkl".format(results_dir,title),'wb') as f:
        pickle.dump(history,f)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
    l = len(history['train_loss'])
    eval_every = history['eval_every']
    ax1.plot(np.arange(l)*eval_every,history['train_loss'],label='Train')
    ax1.plot(np.arange(l)*eval_every,history['dev_loss'],label='Dev')
    ax1.set_title('Loss',fontsize='xx-large')
    ax1.grid(True)
    ax1.legend(loc='upper right',fontsize='x-large')

    ax2.plot(np.arange(l)*eval_every,history['train_accuracy'],label='Train')
    ax2.plot(np.arange(l)*eval_every,history['dev_accuracy'],label='Dev')
    ax2.set_title('Accuracy',fontsize='xx-large')
    ax2.grid(True)
    ax2.legend(loc='lower right',fontsize='x-large')

    fig.tight_layout()
    plt.savefig('{}/{}_history.png'.format(results_dir,title))