import argparse
from datetime import datetime
import os
import pickle

import utils
from model import Classifier

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Random seed:
RANDOM_SEED = 61273812

# Data path:
DATA_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/datav2/'
#DATA_PATH = '/'.join(os.getcwd().split('/')) + '/datav2/'
DATA_ES_PATH = DATA_PATH + 'esp/'
DATA_POR_PATH = DATA_PATH + 'por/'


parser = argparse.ArgumentParser()

# Argumentos para el entrenamiento:
parser.add_argument('--nclasses', type=int, required=True)
parser.add_argument('--lang', type=str, required=True)

# Argumentos para el tokenizer:
parser.add_argument('--pattern', type=str, required=True)
parser.add_argument('--min_ngram', type=int, required=True)
parser.add_argument('--max_ngram', type=int, required=True)
parser.add_argument('--min_df', type=int, required=True)
parser.add_argument('--max_tokens', type=int, required=True)


def validate_args():
    args = parser.parse_args()
    validated_args = {}

    if args.nclasses not in [2,3,5]:
        raise TypeError('Number of classes must be 2, 3 or 5.')
    else:
        validated_args['nclasses'] = args.nclasses

    if args.lang == 'es':
        validated_args['data_path'] = DATA_ES_PATH
        language = 'español'
    elif args.lang == 'pt':
        validated_args['data_path'] = DATA_POR_PATH
        language = 'portugués'
    else:
        raise TypeError('Language must be es or pt')

    if args.min_ngram > args.max_ngram:
        raise TypeError('max_ngram should be greater or equal to min_ngram')
    else:
        ngram_range = (args.min_ngram,args.max_ngram)

    description = """

Descripción del experimento:
----------------------------

Modelo de clasificación con un modelo TfIdf + Naive Bayes.

Argumentos del entrenamiento utilizados:
- Cantidad de clases: {}
- Idioma: {}

Argumentos del tokenizador:
- Patrón para tokenizar: {}
- Frecuencia de documento mínima: {}
- Rango de los n-gramas: {}
- Cantidad máxima de tokens en el vocabulario: {}

""".format(validated_args['nclasses'],language,
    args.pattern,args.min_df,ngram_range,args.max_tokens)

    validated_args['description'] = description

    model_args = {
        'pattern': args.pattern,
        'min_df': args.min_df,
        'ngram_range': ngram_range,
        'max_tokens': args.max_tokens
    }

    return validated_args, model_args


def show_results(y_train_predict,y_train_true,y_dev_predict,y_dev_true,
                 nclasses,description):
    
    now = datetime.now()
    title = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Classification Report:
    report = """
{}

Classification report (train):
------------------------------
    
{}


Classification report (test):
------------------------------
    
{}

    """.format(description,
               classification_report(y_train_true,y_train_predict),
               classification_report(y_dev_true,y_dev_predict))

    with open('results/{}_classification_report.log'.format(title),'w') as f:
        f.write(report)

    # Confusion Matrix:
    cm_train = confusion_matrix(y_train_true,y_train_predict)
    cm_dev = confusion_matrix(y_dev_true,y_dev_predict)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
    im = ax1.imshow(cm_train,cmap='cividis')
    ax1.set_title('Train Confusion Matrix',fontsize='xx-large')
    im = ax2.imshow(cm_dev,cmap='cividis')
    ax2.set_title('Test Confusion Matrix',fontsize='xx-large')

    for ax, cm in [(ax1, cm_train), (ax2, cm_dev)]:
        ticks = list(range(nclasses))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks,fontsize='xx-large')
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks,fontsize='xx-large')
        
        for i in range(nclasses):
            for j in range(nclasses):
                text = ax.text(j, i, cm[i, j],
                            ha="center", va="center", color="red")
    
    fig.tight_layout()
    plt.savefig('results/{}_confusion_matrix.png'.format(title))
    

    
def main(args,model_args):

    # Data loading:
    data_path = args['data_path']
    nclasses = args['nclasses']
    description = args['description']
    print('Loading train data...')
    df_train = utils.load_data(data_path,'train',nclasses)
    df_test = utils.load_data(data_path,'test',nclasses)

    # Model initialization:
    print('Initializing the model...')
    model = Classifier(nclasses,**model_args)

    # Model training:
    print('Training...')
    model.train(df_train['review_content'],
                df_train['review_rate'].values)

    # Model evaluation:
    print('Evaluating results...')
    y_train_predict = model.predict(df_train['review_content'])
    y_train_true = df_train['review_rate'].values
    y_test_predict = model.predict(df_test['review_content'])
    y_test_true = df_test['review_rate'].values
    show_results(y_train_predict,y_train_true,y_test_predict,y_test_true,
                nclasses,description)

    
if __name__ == '__main__':
    args, model_args = validate_args()
    main(args,model_args)
    


