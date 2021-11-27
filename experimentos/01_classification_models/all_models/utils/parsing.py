import argparse
from datetime import datetime
import os
from matplotlib import pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix
import pickle

def parse_args(classifiers):
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--devsize', type=float)
    group.add_argument('--dev', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--model',type=str,required=True)
    parser.add_argument('--dataset',type=str,required=True)

    parser.add_argument('--dropout', type=float, required=False)
    parser.add_argument('--weight_decay', type=float, required=False)
    parser.add_argument('--num_epochs', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=False)
    parser.add_argument('--learning_rate', type=float, required=False)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--eval_every', type=int, required=False)

    args = vars(parser.parse_args())

    dataset = args['dataset']
    dataset, nclasses = dataset.split('-')
    nclasses = int(nclasses)

    dev, test = args['dev'], args['test']
    if dev and not test:
        split = 'dev'
        devsize = 0.
    elif test and not dev:
        split = 'test'
        devsize = 0.
    else:
        split = 'dev'
        devsize = args['devsize']

    dataset_args = dict(
        dataset=dataset,
        nclasses=nclasses,
        split=split,
        devsize=devsize
    )


    model_name = args['model']
    clf_dict = classifiers[model_name]

    clf_cls = clf_dict['clf']
    model_args = clf_dict['model_args']
    hyperparams = clf_dict['hyperparams']
    clf_args = {key:val for key,val in model_args.items()}
    clf_args['nclasses'] = nclasses
    for hyp in hyperparams:
        clf_args[hyp] = args.pop(hyp)
    eval_every = clf_args.pop('eval_every')

    return dict(
        model_name=model_name,
        dataset_args=dataset_args,
        clf_cls=clf_cls,
        clf_args=clf_args,
        eval_every=eval_every
    )



def show_results(y_train_pred,y_train,y_devtest_pred,y_devtest,args,history):

    all_args = {
        **{'model_name':  args['model_name']},
        **args['dataset_args'],
        **args['clf_args']
    }

    calculate_acc = lambda y_pred,y_true: np.mean(y_pred==y_true)*100
    calculate_mae = lambda y_pred,y_true: np.mean(np.abs(y_pred-y_true))*100
    description = """
Args:

{}

Results:

Train Accuracy: {:.2f}%
Test/Dev Accuracy: {:.2f}%
Train MAE: {:.2f}%
Test/Dev MAE: {:.2f}%
    """.format(
        '\n'.join(["{}: {}".format(key,val) for key, val in all_args.items()]),
        calculate_acc(y_train_pred,y_train),
        calculate_acc(y_devtest_pred,y_devtest),
        calculate_mae(y_train_pred,y_train),
        calculate_mae(y_devtest_pred,y_devtest)
    )

    now = datetime.now()
    title = now.strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = os.getcwd() + '/results/{model_name}'.format(**all_args)
    with open('{}/{}_classification_report.log'.format(results_dir,title),'w') as f:
        f.write(description)
    
    nclasses = all_args['nclasses']
    list_of_labels = list(range(nclasses))

    cm_train = confusion_matrix(y_train,y_train_pred,labels=list_of_labels)
    cm_dev = confusion_matrix(y_devtest,y_devtest_pred,labels=list_of_labels)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
    im = ax1.imshow(cm_train,cmap='cividis')
    ax1.set_title('Train Confusion Matrix',fontsize='xx-large')
    im = ax2.imshow(cm_dev,cmap='cividis')
    ax2.set_title('Test/Dev Confusion Matrix',fontsize='xx-large')

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


    with open("{}/{}_history.pkl".format(results_dir,title),'wb') as f:
        pickle.dump(history,f)

    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
    l = len(history['train_loss'])
    eval_every = history['eval_every']
    ax1.plot(np.arange(l)*eval_every,history['train_loss'],label='Train')
    ax1.plot(np.arange(l)*eval_every,history['dev_loss'],label='Dev/Test')
    ax1.set_title('Loss',fontsize='xx-large')
    ax1.grid(True)
    ax1.legend(loc='upper right',fontsize='x-large')

    ax2.plot(np.arange(l)*eval_every,history['train_accuracy'],label='Train')
    ax2.plot(np.arange(l)*eval_every,history['dev_accuracy'],label='Dev/Test')
    ax2.set_title('Accuracy',fontsize='xx-large')
    ax2.grid(True)
    ax2.legend(loc='lower right',fontsize='x-large')

    fig.tight_layout()
    plt.savefig('{}/{}_history.png'.format(results_dir,title))