import argparse
import json
import os
from datetime import datetime
import pickle
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import torch


def create_results_dir(results_root):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    results_dir = str(os.path.join(results_root,timestamp))
    os.mkdir(results_dir)
    return results_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,required=True)
    parser.add_argument('--results_dir',type=str,required=True)
    args = vars(parser.parse_args())
    config_json_file = args['config']
    results_dir = create_results_dir(args['results_dir'])

    with open(config_json_file,'r') as f:
        config = json.load(f)
        
    with open(os.path.join(results_dir,'config.json'),'w') as f:
        json.dump(config,f,indent=4,separators=(',',': '))

    # Dataset args:
    dataset = config['dataset']
    devsize = config['devsize'] if dataset in ["melisa",  "tass"] else 0.0
    dataset_args = dict(dataset=dataset,devsize=devsize)

    # Model args:
    model_kwargs = config['model']

    # Tokenizer args:
    tokenizer_kwargs = config['tokenizer']

    # Train args:
    train_kwargs = config['train_kwargs']

    return_values = dict(
        dataset_args=dataset_args, 
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        train_kwargs=train_kwargs,
        results_dir=results_dir
    )

    return return_values


def plot_history(history,results_dir):
    with open(os.path.join(results_dir,'ppl_history.pkl'), "wb") as f:
        pickle.dump(history,f)

    fig, ax = plt.subplots(1,2,figsize=(10,6))
    train_loss = history['train_loss_history']
    dev_loss = history['dev_loss_history']
    train_f1 = history['train_f1score_history']
    dev_f1 = history['dev_f1score_history']
    train_eval_every = history['train_eval_every']
    dev_eval_every = history['dev_eval_every']
    ax[0].plot(np.arange(len(train_loss))*train_eval_every,train_loss,label='Train')
    ax[0].plot(np.arange(len(dev_loss))*dev_eval_every,dev_loss,label='Dev')
    ax[0].set_title('Loss history',fontsize='xx-large')
    ax[0].grid(True)
    ax[0].legend(loc='upper right',fontsize='x-large')

    ax[1].plot(np.arange(len(train_f1))*train_eval_every,train_f1,label='Train')
    ax[1].plot(np.arange(len(dev_f1))*dev_eval_every,dev_f1,label='Dev')
    ax[1].set_title('f1-score history',fontsize='xx-large')
    ax[1].grid(True)
    ax[1].legend(loc='upper right',fontsize='x-large')
    ax[1].set_ylim(0,100)

    fig.tight_layout()
    plt.savefig(os.path.join(results_dir,'train_dev_history.png'))