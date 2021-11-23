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

EMBEDDINGS_PATH = '/'.join(os.getcwd().split('/')[:-3]) + '/pretrained_models/'
embeddings_file_paths = {
    # https://github.com/dccuchile/spanish-word-embeddings#fasttext-embeddings-from-sbwc
    'fasttext': EMBEDDINGS_PATH + 'fasttext-sbwc.vec',
    'glove': EMBEDDINGS_PATH + 'glove-sbwc.i25.vec',
    'word2vec': EMBEDDINGS_PATH + 'SBW-vectors-300-min5.txt'
}

parser = argparse.ArgumentParser()

# Argumentos para el entrenamiento:
parser.add_argument('--nclasses', type=int, required=True)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--devsize', type=float, required=True)
parser.add_argument('--eval_every', type=int, required=True)

# Argumentos para el tokenizer:
parser.add_argument('--pattern', type=str, required=True)
parser.add_argument('--frequency_cutoff', type=int, required=True)
parser.add_argument('--max_tokens', type=int, required=True)
parser.add_argument('--max_sent_len', type=int, required=True)

# Argumentos para el modelo:
parser.add_argument('--embeddings', type=str, required=True)
parser.add_argument('--filter_sizes', type=str, required=True)
parser.add_argument('--n_filters', type=int, required=True)
parser.add_argument('--dropout', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--device', type=str, required=True)


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

    if args.devsize <= 0:
        raise TypeError('devsize must be greater than 0.')
    else:
        validated_args['devsize'] = args.devsize
    
    if args.eval_every <= 0:
        raise TypeError('eval_every must be greater than 0.')
    else:
        validated_args['eval_every'] = args.eval_every

    if args.embeddings not in embeddings_file_paths.keys():
        raise TypeError('embeddings not supported')
    else:
        embeddings = args.embeddings

    description = """

Descripción del experimento:
----------------------------

Modelo de clasificación con red neuronal convolucional en una dimensión. 
El entrenamiento se hace end-to-end.

Argumentos del entrenamiento utilizados:
- Cantidad de clases: {}
- Idioma: {}
- Proporción utilizada para dev: {}
- Mostrar los datos cada {} batches.

Argumentos del tokenizador:
- Patrón para pretokenizar: {}
- Frecuencia mínima: {}
- Cantidad máxima de tokens en el vocabulario: {}
- Cantidad máxima de tokens por review: {}

Argumentos del modelo:
- Embeddings utilizados: {}
- Tamaño de los filtros: {}
- Cantidad de canales por filtro: {}
- Probabilidad de dropout: {}
- Tamaño del batch: {}
- Tasa de aprendizaje: {}
- Cantidad de epochs: {}
- Dispositivo de entrenamiento: {}

""".format(validated_args['nclasses'],language,validated_args['devsize'],
    validated_args['eval_every'],args.pattern,args.frequency_cutoff,
    args.max_tokens,args.max_sent_len,embeddings,args.filter_sizes,
    args.n_filters,args.dropout,args.batch_size,args.learning_rate,
    args.num_epochs,args.device)

    validated_args['description'] = description

    model_args = {
        'pattern': args.pattern,
        'frequency_cutoff': args.frequency_cutoff,
        'max_tokens': args. max_tokens,
        'max_sent_len': args.max_sent_len,
        'embeddings': embeddings,
        'filter_sizes': [int(fs) for fs in args.filter_sizes.split(',')],
        'n_filters': args.n_filters,
        'dropout': args.dropout,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'device': args.device
    }

    return validated_args, model_args


def show_results(y_train_predict,y_train_true,y_dev_predict,y_dev_true,
                 history,nclasses,description):
    
    now = datetime.now()
    title = now.strftime("%Y-%m-%d-%H-%M-%S")

    # Classification Report:
    report = """
{}

Classification report (train):
------------------------------
    
{}


Classification report (dev):
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
    ax2.set_title('Dev Confusion Matrix',fontsize='xx-large')

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

    # Loss and accuracy history:
    with open("results/{}_history.pkl".format(title),'wb') as f:
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
    plt.savefig('results/{}_history.png'.format(title))

    
def main(args,model_args):

    # Data loading:
    data_path = args['data_path']
    nclasses = args['nclasses']
    dev_size = args['devsize']
    eval_every = args['eval_every']
    description = args['description']
    print('Loading train data...')
    df = utils.load_data(data_path,'train',nclasses)
    df_train, df_dev = utils.train_dev_split(df,dev_size=dev_size,random_state=RANDOM_SEED)

    # Model initialization:
    print('Initializing the model...')
    model = Classifier(nclasses,**model_args)

    # Model training:
    print('Training...')
    history = model.train(
                            df_train['review_content'],
                            df_train['review_rate'].values,
                            eval_every=eval_every,
                            dev=(
                               df_dev['review_content'],
                               df_dev['review_rate'].values
                            )
                        )

    # Model evaluation:
    print('Evaluating results...')
    y_train_predict = model.predict(df_train['review_content'])
    y_train_true = df_train['review_rate'].values
    y_dev_predict = model.predict(df_dev['review_content'])
    y_dev_true = df_dev['review_rate'].values
    show_results(y_train_predict,y_train_true,
                 y_dev_predict,y_dev_true,
                 history,nclasses,description)

    
if __name__ == '__main__':
    args, model_args = validate_args()
    main(args,model_args)
    


