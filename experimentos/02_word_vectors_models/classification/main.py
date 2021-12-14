import pickle
from matplotlib import pyplot as plt
import numpy as np
from utils import WordTokenizer, ElmoTokenizer, parse_args, plot_history
from models import *
import os
import torch
from torch import nn, optim
from utils.data import load_melisa, load_and_split_melisa, \
                    load_amazon, load_tass, load_and_split_tass, \
                    load_cine, normalize_dataset
import pandas as pd
from time import time
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm

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
    
    else:
        raise NameError("Dataset not supported")
            
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

    if embeddings is None:
        embeddings_path = EMBEDDINGS_ROOT_PATH
        embedding_dim = model_kwargs.pop("embedding_dim")
        embedding = WordEmbedding(embeddings=None,tokenizer=tokenizer,
                            embeddings_path=embeddings_path,embedding_dim=embedding_dim)
    elif embeddings in ["word2vec300", "glove300"]:
        embeddings_path = EMBEDDINGS_ROOT_PATH
        embedding = WordEmbedding(embeddings=embeddings,tokenizer=tokenizer,
                            embeddings_path=embeddings_path,embedding_dim=None)
    elif embeddings == "fasttext300":
        embeddings_path = EMBEDDINGS_ROOT_PATH
        embedding = FastTextEmbedding(embeddings,tokenizer,embeddings_path)
    elif embeddings == "elmo":
        embeddings_path = os.path.join(EMBEDDINGS_ROOT_PATH,"elmo/")
        embedding = ELMOEmbedding(tokenizer,embeddings_path,model_kwargs.pop("elmo_batch_size"))
    else:
        raise NameError("Embeddings not supported")

    if model_name == "cbow":
        model = CBOWClassifier(embedding,num_outs=5,**model_kwargs)
    elif model_name == "lstm":
        model = RNNClassifier(embedding,rnn="LSTM",num_outs=5,num_layers=1,bidirectional=True,**model_kwargs)
    elif model_name == "cnn":
        model = CNNClassifier(embedding,nclasses=5,**model_kwargs)
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
    return f1_score(y_true,y_pred,average="macro")*100


def evaluate_f1_loss_dev(model,sent_dev,y_dev,batch_size,criterion,device):

    was_training = model.training
    model.eval()

    cum_loss = cum_num_examples = 0
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
            num_examples = len(batch_sent)
            cum_num_examples += num_examples

    total_loss = cum_loss / cum_num_examples
    y_true = np.hstack(cum_dev_labels_true)
    y_pred = np.hstack(cum_dev_labels_pred)
    f1 = f1_score(y_true,y_pred,average="macro")*100

    if was_training:
        model.train()

    return total_loss, f1


def save_checkpoint(model,optimizer,path):
    params = {
        'model_state_dict': model.state_dict(),
        'tokenizer': model.emb.tokenizer,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(params,path)

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
    criterion = nn.CrossEntropyLoss(reduction='sum')

    model.to(device)

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
            num_examples = len(batch_sent)
            cum_train_loss += loss_val
            cum_num_examples += num_examples
            cum_train_labels_pred.append(scores.cpu().detach().max(dim=1).indices.numpy())
            cum_train_labels_true.append(batch_y.cpu().numpy())

            if (e * num_batches + i) % train_eval_every == 0:
                print('Batch {}/{}. Epoch {}/{}'.format(i,num_batches,e+1,max_epochs))
                avg_loss = cum_train_loss / cum_num_examples
                print('Train avg. loss: {:.5f}'.format(avg_loss))
                train_loss_history.append(avg_loss)
                avg_f1 = evaluate_f1_train(cum_train_labels_true,cum_train_labels_pred)
                print('Train avg. f1-score : {:.5f}'.format(avg_f1))
                train_f1score_history.append(avg_f1)
                cum_train_loss = cum_num_examples = 0
                cum_train_labels_pred = []
                cum_train_labels_true = []

            if (e * num_batches + i) % dev_eval_every == 0:
                print('Dev evaluation...')
                dev_loss, dev_f1 = evaluate_f1_loss_dev(model,data['sent_dev'],data['y_dev'],dev_batch_size,criterion,device)
                print('Dev loss: {:.5f}. Dev f1-score: {:.5f}'.format(dev_loss,dev_f1))

                is_better = len(dev_f1score_history) == 0 or dev_f1 > max(dev_f1score_history)
                dev_f1score_history.append(dev_f1)
                dev_loss_history.append(dev_loss)

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
    training_time = end_time-start_time
    print("Duración del entrenamiento: {} horas {} minutos {} segundos".format(
                                                        int(training_time//3600),
                                                        int((training_time % 3600) // 60),
                                                        int(training_time % 60) 
                                                    )
    )
    history = dict(
        train_loss_history=train_loss_history,
        train_f1score_history=train_f1score_history,
        dev_loss_history=dev_loss_history,
        dev_f1score_history=dev_f1score_history,
        train_eval_every=train_eval_every,
        dev_eval_every=dev_eval_every,
        max_epochs=max_epochs,
        last_epoch=e
    )
    return history
    

def evaluate_all(model,data,results_dir,history,**kwargs):

    print("Loading best model for evaluation...")
    state_dict = torch.load(os.path.join(results_dir,"checkpoint.pkl"))['model_state_dict']
    model.load_state_dict(state_dict)

    model.eval()
    device = torch.device(kwargs['device'])
    criterion = nn.CrossEntropyLoss(reduction='sum')

    results = {}
    for split in ["train", "dev"]:

        cum_loss = 0
        cum_labels_pred = []
        cum_labels_true = []

        num_batches = len(torch.arange(len(data['y_'+split])).split(kwargs[split+'_batch_size']))

        with torch.no_grad():
            for i, batch in tqdm(batch_iter(
                                data['sent_'+split],
                                data['y_'+split],
                                kwargs[split+'_batch_size'],
                                device,
                                shuffle=False
                            ),total=num_batches):

                batch_sent, batch_y = batch
                scores = model(batch_sent)
                loss = criterion(scores,batch_y)

                cum_labels_pred.append(scores.cpu().max(dim=1).indices.numpy())
                cum_labels_true.append(batch_y.cpu().numpy())

                cum_loss += loss.item()

        total_loss = cum_loss / len(data['y_'+split])
        y_true = np.hstack(cum_labels_true)
        y_pred = np.hstack(cum_labels_pred)
        f1 = f1_score(y_true,y_pred,average="macro")*100

        print()
        print("{} RESULTS:".format(split.upper()))
        print("-----------")
        print("Avg loss: {:.3f}".format(total_loss))
        print("f1-score macro (%): {:.2f}".format(f1))

        results[split] = dict(
            loss=total_loss,
            f1=f1,
            y_true=y_true,
            y_pred=y_pred
        )

    with open(os.path.join(results_dir,"results.pkl"),"wb") as f:
        pickle.dump(results,f)

    with open(os.path.join(results_dir,"results.txt"),"w") as f:
        f.write("""

Epochs: {}/{}
        
TRAIN RESULTS:
--------------
Avg loss: {:.3f}
MAE (x100): {:.2f}

Classification Report:

{}


DEV RESULTS:
------------
Avg loss: {:.3f}
MAE (x100): {:.2f}

Classification Report:

{}
        
        """.format(
            history['max_epochs'],history['last_epoch'],
            results['train']['loss'],
            calculate_mae(results['train']['y_true'],results['train']['y_pred']),
            classification_report(results['train']['y_true'],results['train']['y_pred'],digits=6),
            results['dev']['loss'],
            calculate_mae(results['dev']['y_true'],results['dev']['y_pred']),
            classification_report(results['dev']['y_true'],results['dev']['y_pred'],digits=6)
            )
        )

    list_of_labels = list(range(5))

    cm_train = confusion_matrix(results['train']['y_true'],results['train']['y_pred'],labels=list_of_labels)
    cm_dev = confusion_matrix(results['dev']['y_true'],results['dev']['y_pred'],labels=list_of_labels)
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,6))
    im = ax1.imshow(cm_train,cmap='cividis')
    ax1.set_title('Train Confusion Matrix',fontsize='xx-large')
    im = ax2.imshow(cm_dev,cmap='cividis')
    ax2.set_title('Dev Confusion Matrix',fontsize='xx-large')

    for ax, cm in [(ax1, cm_train), (ax2, cm_dev)]:
        ticks = list_of_labels
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks,fontsize='xx-large')
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks,fontsize='xx-large')
        
        for i in list_of_labels:
            for j in list_of_labels:
                text = ax.text(j, i, cm[i, j],
                            ha="center", va="center", color="red")
    
    fig.tight_layout()
    plt.savefig(os.path.join(results_dir,'confusion_matrix.png'))



def calculate_mae(y_true,y_pred):
    return np.abs(y_true-y_pred).mean()*100

def main():

    # Read args
    print("Parsing args...")
    args = parse_args()
        
    # Dataset loading:
    print("Loading dataset...")
    dataset_args = args.pop('dataset_args')
    data = load_dataset(nclasses=5,**dataset_args)
    # data = {key: val[:1000] for key, val in data.items()}

    # Inicialización del tokenizer
    print("Initializing tokenizer...")
    tokenizer_kwargs = args.pop('tokenizer_kwargs')
    tokenizer = init_tokenizer(data['sent_train'],tokenizer_kwargs)
    
    # Inicialización del modelo
    print("Initializing model...")
    model_kwargs = args.pop('model_kwargs')
    model = init_model(model_kwargs,tokenizer)
    print("Training...")
    history = train(
        model=model,
        data=data,
        results_dir=args['results_dir'],
        **args['train_kwargs']
    )

    # Gráfico del historial
    print("Plotting history...")
    plot_history(history,args['results_dir'])

    # Evaluación de los resultados
    print("Evaluating results...")
    evaluate_all(model,data,args['results_dir'],history,**args['train_kwargs'])

    


if __name__ == "__main__":
    main()