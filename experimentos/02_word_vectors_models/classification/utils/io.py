import argparse
import json
import os
from datetime import datetime
import pickle
from copy import deepcopy
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
    devsize = config['devsize'] if dataset == "melisa" else 0.0
    dataset_args = dict(dataset=dataset,devsize=devsize)

    # Model args:
    model_kwargs = config['model']

    # Tokenizer args:
    tokenizer_kwargs = config['tokenizer']

    # Train args:
    train_kwargs = config['train_kwargs']

    # Eval args:
    eval_kwargs = config['eval_kwargs']

    return_values = dict(
        dataset_args=dataset_args, 
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs,
        train_kwargs=train_kwargs,
        eval_kwargs=eval_kwargs,
        results_dir=results_dir
    )

    return return_values