from datetime import datetime
import os
import pickle
from time import time
import numpy as np
from torch import optim
import utils
from utils.data import load_melisa, load_amazon, load_and_split_melisa, normalize_dataset
from utils.io import parse_args
from models import init_lstm_model, WordTokenizer, greedy_decoding
import matplotlib.pyplot as plt
from tqdm import tqdm


import pandas as pd
import torch


def load_data(dataset,devsize):
    print('Loading dataset...')
    if dataset == 'melisa':
            df_train, df_dev = load_and_split_melisa(devsize)
    elif dataset == 'amazon':
            df_train = load_amazon(split='train')
            df_dev = load_amazon(split='dev')
    else:
        raise NameError('Dataset not supported')

    print('Normalizing dataset...')
    #n = 1000
    ds_src_train = normalize_dataset(df_train['review_content'])#.iloc[:n])
    ds_target_train = normalize_dataset(df_train['review_title'])#.iloc[:n])
    ds_src_dev = normalize_dataset(df_dev['review_content'])#.iloc[:n])
    ds_target_dev = normalize_dataset(df_dev['review_title'])#.iloc[:n])
    data = {
        'src_train': ds_src_train, 
        'tgt_train': ds_target_train, 
        'src_dev': ds_src_dev, 
        'tgt_dev': ds_target_dev
    }
    return data


def init_model(model_kwargs,src_tokenizer_kwargs,tgt_tokenizer_kwargs):

    model_name = model_kwargs.pop('name')
    src_tokenizer = init_tokenizer(**src_tokenizer_kwargs)
    tgt_tokenizer = init_tokenizer(**tgt_tokenizer_kwargs)
    
    if model_name == "lstm":
        model = init_lstm_model(src_tokenizer,tgt_tokenizer,**model_kwargs)
    elif model_name == "transformer":
        pass
    else:
        raise NameError("Model not supported")

    return model, src_tokenizer, tgt_tokenizer


def init_tokenizer(**tokenizer_kwargs):

    tokenizer_name = tokenizer_kwargs.pop('name')

    if tokenizer_name == "word_tokenizer":
        tokenizer = WordTokenizer.from_dataseries(**tokenizer_kwargs)
    elif tokenizer_name == "transformer":
        pass
        # tokenizer = BertTokenizer.from_pretrained(src_path,**tokenizer_kwargs)
    else:
        raise NameError("Model not supported")
    
    return tokenizer


def batch_iter(src_sents,tgt_sents,src_tokenizer,tgt_tokenizer,batch_size,shuffle=True):

    df = pd.concat((src_sents,tgt_sents),keys=['src','tgt'],axis=1)
    N = len(df)

    if shuffle:
        indices_batches = torch.randperm(N).split(batch_size)
    else:
        indices_batches = torch.arange(N).split(batch_size)

    for i, indices in enumerate(indices_batches):
        batch = df.iloc[indices,:]
        src_encoded_input = src_tokenizer(batch['src'])
        tgt_encoded_input = tgt_tokenizer(batch['tgt'])

        _, indices = src_encoded_input['attention_mask'].sum(dim=1).sort(descending=True)
        src_encoded_input['input_ids'] = src_encoded_input['input_ids'][indices,:]
        src_encoded_input['attention_mask'] = src_encoded_input['attention_mask'][indices,:]
        tgt_encoded_input['input_ids'] = tgt_encoded_input['input_ids'][indices,:]
        tgt_encoded_input['attention_mask'] = tgt_encoded_input['attention_mask'][indices,:]
        
        yield i, (src_encoded_input, tgt_encoded_input)


def train_model(
        model,
        data,
        src_tokenizer,
        tgt_tokenizer,
        results_dir,
        **kwargs
    ):

    max_epochs = kwargs.pop('max_epochs')
    lr = kwargs.pop('learning_rate')
    train_batch_size = kwargs.pop('train_batch_size')
    dev_batch_size = kwargs.pop('dev_batch_size')
    train_eval_every = kwargs.pop('train_eval_every')
    dev_eval_every = kwargs.pop('dev_eval_every')
    patience = kwargs.pop('patience')
    checkpoint_path = str(os.path.join(results_dir,'checkpoint.pkl'))
    
    optimizer = optim.Adam(model.parameters(),lr=lr)

    train_ppl_history = []
    dev_ppl_history = []
    
    num_batches = len(torch.arange(len(data['src_train'])).split(train_batch_size))
    cum_train_ppl = cum_num_examples = total_words_to_predict = patience_count = e = 0
    start_time = time()
    while e < max_epochs:
        for i, batch in batch_iter(
                            data['src_train'], data['tgt_train'],
                            src_tokenizer,tgt_tokenizer,
                            train_batch_size,
                            shuffle=True
                        ):

            src_encoded_input, tgt_encoded_input = batch

            optimizer.zero_grad()
            _, loss = model(src_encoded_input,tgt_encoded_input)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            num_examples = src_encoded_input['input_ids'].size(0)
            cum_train_ppl += loss_val
            cum_num_examples += num_examples
            total_words_to_predict += src_encoded_input['attention_mask'][:,1:].sum().item()

            if (e * num_batches + i) % train_eval_every == 0:
                print('Batch {}/{}. Epoch {}/{}'.format(i,num_batches,e+1,max_epochs))
                avg_ppl = np.exp(cum_train_ppl / total_words_to_predict)
                print('Train avg. ppl: {:.5f}'.format(avg_ppl))
                train_ppl_history.append(avg_ppl)
                cum_train_ppl = cum_num_examples = total_words_to_predict = 0

            if (e * num_batches + i) % dev_eval_every == 0:
                print('Dev evaluation...')
                dev_ppl = evaluate_ppl(model,data['src_dev'],data['tgt_dev'],src_tokenizer,tgt_tokenizer,dev_batch_size)
                print('Dev avg. ppl: {:.5f}'.format(dev_ppl))

                is_better = len(dev_ppl_history) == 0 or dev_ppl < min(dev_ppl_history)
                dev_ppl_history.append(dev_ppl)

                if is_better:
                    patience_count = 0
                    save_checkpoint(model,optimizer,checkpoint_path)
                else:
                    patience_count += 1
                    print("Valor de la paciencia: #{}".format(patience_count))
                    if patience_count == patience:
                        break
        
        if patience_count == patience:
            print("Se agotó la paciencia!")
            break

        e += 1

    end_time = time()
    print("Duración del entrenamiento: {:.2f} horas".format((end_time-start_time) / 3600))
    history = dict(
        train_history=train_ppl_history,
        dev_history=dev_ppl_history,
        train_eval_every=train_eval_every,
        dev_eval_every=dev_eval_every,

    )
    return history
    

def plot_and_save_ppl(history,results_dir):
    with open(os.path.join(results_dir,'ppl_history.pkl'), "wb") as f:
        pickle.dump(history,f)

    fig, ax = plt.subplots(1,1,figsize=(10,6))
    train_ppl = history['train_history']
    dev_ppl = history['dev_history']
    train_eval_every = history['train_eval_every']
    dev_eval_every = history['dev_eval_every']
    ax.plot(np.arange(len(train_ppl))*train_eval_every,train_ppl,label='Train Perplexity')
    ax.plot(np.arange(len(dev_ppl))*dev_eval_every,dev_ppl,label='Dev Perplexity')
    ax.set_title('Perplexity history',fontsize='xx-large')
    ax.grid(True)
    ax.set_yscale('log')
    ax.legend(loc='upper right',fontsize='x-large')

    fig.tight_layout()
    plt.savefig(os.path.join(results_dir,'ppl_history.png'))



def save_checkpoint(model,optimizer,path):
    params = {
        'model_state_dict': model.state_dict(),
        'src_tokenizer': model.encoder.tokenizer,
        'tgt_tokenizer': model.decoder.tokenizer,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(params,path)



def evaluate_ppl(model,src_dev,tgt_dev,src_tokenizer,tgt_tokenizer,batch_size=32):
    
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    with torch.no_grad():
        for i, (src_encoded_input, tgt_encoded_input) in batch_iter(
                            src_dev, tgt_dev,
                            src_tokenizer,tgt_tokenizer,
                            batch_size,
                            shuffle=False
                        ):
            _, loss = model(src_encoded_input,tgt_encoded_input)

            cum_loss += loss.item()
            tgt_word_num_to_predict = tgt_encoded_input['attention_mask'][:,1:].sum().item() # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def save_sents_to_file(tgt_sents,pred_sents,path):

    df = pd.concat([
        pd.Series(tgt_sents),
        pd.Series(pred_sents)
    ], ignore_index=True, axis=1)
    df = df.rename(columns={0: "True", 1: "Predicted"})
    df.to_csv(path,index=False)



def main():

    # Read args
    args = parse_args()
        
    # Dataset loading:
    dataset_args = args.pop('dataset_args')
    data = load_data(**dataset_args)

    # Model Initialization
    model_kwargs = args.pop('model_kwargs')
    src_tokenizer_kwargs = args.pop('src_tokenizer_kwargs')
    tgt_tokenizer_kwargs = args.pop('tgt_tokenizer_kwargs')
    
    if (src_tokenizer_kwargs['name'] == 'word_tokenizer') or \
        (tgt_tokenizer_kwargs['name'] == 'word_tokenizer'):
        ds = pd.concat([data['src_train'],data['tgt_train']],ignore_index=True)
    if src_tokenizer_kwargs['name'] == 'word_tokenizer':
        src_tokenizer_kwargs['ds'] = ds
    if tgt_tokenizer_kwargs['name'] == 'word_tokenizer':
        tgt_tokenizer_kwargs['ds'] = ds

    model, src_tokenizer, tgt_tokenizer = init_model(
        model_kwargs,
        src_tokenizer_kwargs,
        tgt_tokenizer_kwargs
    )

    # Training
    history = train_model(
        model=model,
        data=data,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        results_dir=args['results_dir'],
        **args['train_kwargs']
    )

    plot_and_save_ppl(history,args['results_dir'])

    print("Evaluating results...")
    eval_splits = args['eval_kwargs'].pop('eval_in')
    max_len_of_pred_sent = args['eval_kwargs'].pop('max_len_of_pred_sent')
    eval_data = {}
    if "train" in eval_splits:
        eval_data['train'] = (data.pop("src_train"),data.pop("tgt_train"))
    if "dev" in eval_splits:
        eval_data['dev'] = (data.pop("src_dev"),data.pop("tgt_dev"))
    
    for split, (src_sents, target_sents) in eval_data.items():
        pred_sents = greedy_decoding(model,src_sents,tgt_tokenizer,max_len_of_pred_sent)
        sents_path_file = os.path.join(args['results_dir'],"{}_results.csv".format(split))
        target_sents = [tgt_tokenizer.tokenize(sent) for sent in target_sents]
        save_sents_to_file(target_sents,pred_sents,sents_path_file)



if __name__ == '__main__':
    main()