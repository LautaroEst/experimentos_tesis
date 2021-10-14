import argparse
from datetime import datetime
import os

import utils
from model import Classifier

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Random seed:
RANDOM_SEED = 61273812

# Data path:
DATA_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/datav2/'
DATA_ES_PATH = DATA_PATH + 'esp/'
DATA_POR_PATH = DATA_PATH + 'por/'


parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--devsize', type=float, required=True)
parser.add_argument('--description', type=str, required=True)


def validate_args():
    args = parser.parse_args()
    validated_args = {}

    if args.nclasses not in [2,3,5]:
        raise TypeError('Number of classes must be 2, 3 or 5.')
    else:
        validated_args['nclasses'] = args.nclasses

    if args.lang == 'es':
        validated_args['data_path'] = DATA_ES_PATH
    elif args.lang == 'pt':
        validated_args['data_path'] = DATA_POR_PATH
    else:
        raise TypeError('Language must be es or pt')

    validated_args['devsize'] = args.devsize
    validated_args['description'] = args.description

    return validated_args


def show_results(y_predict,y_true,nclasses,title,description):
    
    report = """

Descripción del experimento:
----------------------------

{}


Classification report:
----------------------
    
{}

    """.format(description,classification_report(y_true,y_predict))

    with open('results/{}_classification_report.log'.format(title),'w') as f:
        f.write(report)

    cm = confusion_matrix(y_true,y_predict)

    fig, ax = plt.subplots(1,1,figsize=(10,6))
    im = ax.imshow(cm,cmap='cividis')

    ax.set_xticks(np.arange(nclasses))
    ax.set_yticks(np.arange(nclasses))
    ax.set_xticklabels(np.arange(nclasses),fontsize='xx-large')
    ax.set_yticklabels(np.arange(nclasses),fontsize='xx-large')

    
    # Loop over data dimensions and create text annotations.
    for i in range(nclasses):
        for j in range(nclasses):
            text = ax.text(j, i, cm[i, j],
                        ha="center", va="center", color="red")

    ax.set_title('Confusion Matrix',fontsize='xx-large')
    fig.tight_layout()
    plt.savefig('results/{}_confusion_matrix.png'.format(title))

    
def main(args):

    # Data loading:
    data_path = args['data_path']
    nclasses = args['nclasses']
    dev_size = args['devsize']
    description = args['description']
    print('Loading train data...')
    df = utils.load_data(data_path,'train',nclasses)
    if dev_size == 0:
        df_train = df
    else:
        df_train, df_dev = utils.train_dev_split(df,dev_size=dev_size,random_state=RANDOM_SEED)

    # Model initialization:
    print('Initializing the model...')
    model = Classifier()

    # Model training:
    print('Training...')
    model.train(df_train['review_content'],df_train['review_rate'].values)

    # Model evaluation:
    print('Evaluating results on train...')
    now = datetime.now()
    train_title = now.strftime("%Y-%m-%d-%H-%M-%S_train")
    y_predict = model.predict(df_train['review_content'])
    y_true = df_train['review_rate'].values
    show_results(y_predict,y_true,nclasses,train_title,description)

    if dev_size > 0:
        print('Evaluating results on dev...')
        dev_title = now.strftime("%Y-%m-%d-%H-%M-%S_dev")
        y_predict = model.predict(df_dev['review_content'])
        y_true = df_dev['review_rate'].values
        show_results(y_predict,y_true,nclasses,dev_title,description)



if __name__ == '__main__':
    args = validate_args()
    main(args)
    


