import pickle
from matplotlib import pyplot as plt
import numpy as np
from utils import WordTokenizer, ElmoTokenizer, parse_args
from models import *
import os
import torch
from torch import nn, optim
from utils.data import load_melisa, load_and_split_melisa, \
                    load_amazon, load_tass, load_and_split_tass, \
                    load_cine, normalize_dataset
import pandas as pd
from time import time
from sklearn.metrics import f1_score

EMBEDDINGS_ROOT_PATH = os.path.join(os.getcwd(),"../../../pretrained_models/")


def load_dataset(nclasses,dataset,devsize):
    if dataset == 'melisa':
        df_train, df_dev = load_and_split_melisa(nclasses,devsize)
        ds_train = normalize_dataset(df_train['review_content'])
        ds_dev = normalize_dataset(df_dev['review_content'])
        y_train = df_train['review_rate']
        y_dev = df_dev['review_rate']
        
    elif dataset == 'amazon':
        df_train = load_amazon(split='train',nclasses=nclasses)
        df_dev = load_amazon(split='dev',nclasses=nclasses)
        ds_train = normalize_dataset(df_train['review_content'])
        ds_dev = normalize_dataset(df_dev['review_content'])
        y_train = df_train['review_rate']
        y_dev = df_dev['review_rate']

    elif dataset == 'tass':
        df_train, df_dev = load_and_split_tass(nclasses,devsize)
        ds_train = normalize_dataset(df_train['tweet'])
        ds_dev = normalize_dataset(df_dev['tweet'])
        y_train = df_train['label']
        y_dev = df_dev['label']

    elif dataset == 'cine':
        df_train = load_cine(split='train',nclasses=nclasses)
        df_dev = load_cine(split='dev',nclasses=nclasses)

        def limit_len(ds):
            ds = ds.apply(lambda s: s[:2000])
            return ds

        ds_train = limit_len(normalize_dataset(df_train['review_content']))
        ds_dev = limit_len(normalize_dataset(df_dev['review_content']))
        y_train = df_train['review_rate']
        y_dev = df_dev['review_rate']
            
    data = {
        'sent_train': ds_train, 
        'y_train': y_train, 
        'sent_dev': ds_dev, 
        'y_dev': y_dev
    }
    return data


def init_model(model_kwargs,tokenizer):

    model_name = model_kwargs.pop("name")
    embeddings = model_kwargs.pop("embeddings")

    if embeddings is None or embeddings in ["word2vec300", "glove300"]:
        embeddings_path = EMBEDDINGS_ROOT_PATH
        embedding = WordEmbedding(embeddings,tokenizer,embeddings_path,**model_kwargs)
    elif embeddings == "fasttext300":
        embeddings_path = EMBEDDINGS_ROOT_PATH
        embedding = FastTextEmbedding(tokenizer,embeddings_path)
    elif embeddings == "elmo":
        embeddings_path = os.path.join(EMBEDDINGS_ROOT_PATH,"elmo/")
        embedding = ELMOEmbedding(tokenizer,embeddings_path,model_kwargs.pop("elmo_batch_size"))
    else:
        raise NameError("Embeddings not supported")

    if model_name == "cbow":
        model = CBOWClassifier(embedding,num_outs=5,**model_kwargs)
    elif model_name == "lstm":
        model = None
    elif model_name == "gru":
        model = None
    elif model_name == "cnn":
        model = None
    else:
        raise NameError("Model not supported")

    return model


def init_tokenizer(data,tokenizer_kwargs):
    tokenizer_name = tokenizer_kwargs.pop("name")
    if tokenizer_name == "elmo_tokenizer":
        tokenizer = ElmoTokenizer(**tokenizer_kwargs)
    elif tokenizer_name == "word_tokenizer":
        tokenizer = WordTokenizer.from_dataseries(data,**tokenizer_kwargs)
    else:
        raise NameError("Tokenizer is not supported.")

    return tokenizer

def batch_iter(sent,y,batch_size,device,shuffle=True):

    df = pd.concat((sent,y),keys=['x','y'],axis=1)
    N = len(df)

    if shuffle:
        indices_batches = torch.randperm(N).split(batch_size)
    else:
        indices_batches = torch.arange(N).split(batch_size)

    for i, indices in enumerate(indices_batches):
        batch = df.iloc[indices,:].reset_index(drop=True)
        sents_batch = batch['x']
        y_batch = torch.from_numpy(batch['y'].values).to(device=device)

        yield i, (sents_batch, y_batch)


def evaluate_f1_train(y_true_parts,y_pred_parts):
    y_true = np.hstack(y_true_parts)
    y_pred = np.hstack(y_pred_parts)
    return f1_score(y_true,y_pred)


def evaluate_f1_loss_dev(model,sent_dev,y_dev,batch_size,criterion,device):

    was_training = model.training
    model.eval()

    cum_loss = 0
    cum_dev_labels_pred = []
    cum_dev_labels_true = []

    with torch.no_grad():
        for i, batch in batch_iter(
                            sent_dev,
                            y_dev,
                            batch_size,
                            device,
                            shuffle=False
                        ):

            batch_sent, batch_y = batch
            scores = model(batch_sent)
            loss = criterion(scores,batch_y)

            cum_dev_labels_pred.append(scores.cpu().max(dim=1).indices.numpy())
            cum_dev_labels_true.append(batch_y.cpu().numpy())

            cum_loss += loss.item()

    total_loss = cum_loss / len(y_dev)
    y_true = np.hstack(cum_dev_labels_true)
    y_pred = np.hstack(cum_dev_labels_pred)
    f1 = f1_score(y_true,y_pred)

    if was_training:
        model.train()

    return total_loss, f1


def save_checkpoint(model,optimizer,path):
    params = {
        'model_state_dict': model.state_dict(),
        'src_tokenizer': model.encoder.tokenizer,
        'tgt_tokenizer': model.decoder.tokenizer,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(params,path)


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


def train(
        model,
        data,
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
    device = torch.device(kwargs.pop('device'))
    checkpoint_path = str(os.path.join(results_dir,'checkpoint.pkl'))
    
    optimizer = optim.Adam(model.parameters(),lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    train_f1score_history = []
    dev_loss_history = []
    dev_f1score_history = []
    
    num_batches = len(torch.arange(len(data['sent_train'])).split(train_batch_size))
    cum_train_loss = cum_num_examples = patience_count = e = 0
    cum_train_labels_pred = []
    cum_train_labels_true = []
    start_time = time()
    while e < max_epochs:
        for i, batch in batch_iter(
                            data['sent_train'],
                            data['y_train'],
                            train_batch_size,
                            device,
                            shuffle=True
                        ):

            batch_sent, batch_y = batch

            optimizer.zero_grad()
            scores = model(batch_sent)
            loss = criterion(scores,batch_y)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            num_examples = batch_sent.size(0)
            cum_train_loss += loss_val
            cum_num_examples += num_examples
            cum_train_labels_pred.append(scores.cpu().detach().max(dim=1).indices.numpy())
            cum_train_labels_true.append(batch_y.cpu().numpy())

            if (e * num_batches + i) % train_eval_every == 0:
                print('Batch {}/{}. Epoch {}/{}'.format(i,num_batches,e+1,max_epochs))
                avg_loss = cum_train_loss / cum_num_examples
                print('Train avg. loss: {:.5f}'.format(avg_loss))
                train_loss_history.append(avg_loss)
                train_f1 = evaluate_f1_train(cum_train_labels_true,cum_train_labels_pred)
                print('Train f1-score : {:.5f}'.format(avg_loss))
                train_f1score_history.append(train_f1)
                cum_train_loss = cum_num_examples = 0
                cum_train_labels_pred = []
                cum_train_labels_true = []

            if (e * num_batches + i) % dev_eval_every == 0:
                print('Dev evaluation...')
                dev_loss, dev_f1 = evaluate_f1_loss_dev(model,data['sent_dev'],data['y_dev'],dev_batch_size,criterion,device)
                print('Dev loss: {:.5f}. Dev f1-score: {:.5f}'.format(dev_loss,dev_f1))

                is_better = len(dev_f1score_history) == 0 or dev_f1 > max(dev_f1score_history)
                dev_f1score_history.append(dev_f1score_history)

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
        train_loss_history=train_loss_history,
        train_f1score_history=train_f1score_history,
        dev_loss_history=dev_loss_history,
        dev_f1score_history=dev_f1score_history,
        train_eval_every=train_eval_every,
        dev_eval_every=dev_eval_every,
    )
    return history
    


def main():

    # Read args
    args = parse_args()
        
    # Dataset loading:
    dataset_args = args.pop('dataset_args')
    data = load_dataset(nclasses=5,**dataset_args)
    data = {key: val[:1000] for key, val in data.items()}

    # Inicialización del tokenizer
    tokenizer_kwargs = args.pop('tokenizer_kwargs')
    tokenizer = init_tokenizer(data['sent_train'],tokenizer_kwargs)
    
    # Inicialización del modelo
    model_kwargs = args.pop('model_kwargs')
    model = init_model(model_kwargs,tokenizer)
    history = train(
        model=model,
        data=data,
        results_dir=args['results_dir'],
        **args['train_kwargs']
    )

    


if __name__ == "__main__":
    main()