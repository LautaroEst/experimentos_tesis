import argparse
import json
import os
from datetime import datetime
import pickle
from copy import deepcopy
from models import init_lstm_model

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
    src_tokenizer_kwargs = config['src_tokenizer']
    tgt_tokenizer_kwargs = config['tgt_tokenizer']

    # Train args:
    train_kwargs = config['train_kwargs']

    # Eval args:
    eval_kwargs = config['eval_kwargs']

    return_values = dict(
        dataset_args=dataset_args, 
        model_kwargs=model_kwargs,
        src_tokenizer_kwargs=src_tokenizer_kwargs,
        tgt_tokenizer_kwargs=tgt_tokenizer_kwargs,
        train_kwargs=train_kwargs,
        eval_kwargs=eval_kwargs,
        results_dir=results_dir
    )

    return return_values


def read_results(results_dir):

    with open(os.path.join(results_dir,"checkpoint.pkl"),"rb") as f:
        checkpoint = torch.load(f)
    with open(os.path.join(results_dir,"config.json"),"rb") as f:
        train_configurations = json.load(f)
    with open(os.path.join(results_dir,"ppl_history.pkl"),"rb") as f:
        history = pickle.load(f)
    
    results = dict(
        model_state_dict=checkpoint['model_state_dict'],
        optimizer_state_dict=checkpoint['optimizer_state_dict'],
        src_tokenizer=checkpoint['src_tokenizer'],
        tgt_tokenizer=checkpoint['tgt_tokenizer'],
        config=train_configurations,
        history=history,
        results_dir=results_dir
    )

    return results


def load_model(results):

    config = deepcopy(results['config'])
    model_kwargs = config['model']
    model_name = model_kwargs.pop('name')
    if model_name == "lstm":
        model = init_lstm_model(results['src_tokenizer'],results['tgt_tokenizer'],**model_kwargs)
        model.load_state_dict(results['model_state_dict'])
        model.encoder.cuda(0)
        model.decoder.cuda(1)
    elif model_name == "transformer":
        pass
    else:
        raise NameError("Model not supported")

    return model