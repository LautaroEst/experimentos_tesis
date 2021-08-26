import numpy as np
import re
import torch
import time


def get_train_dev_idx(N,dev_size=.2,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    if dev_size == 0:
        return rand_idx

    N_train = int(N * (1-dev_size))
    if N_train == N:
        print('Warning: dev_size too small!')
        N_train = N-1
    
    return rand_idx[:N_train], rand_idx[N_train:]


def train_dev_split(df,random_state=0,dev_size=0.1):
    N_data = len(df)
    train_idx, dev_idx = get_train_dev_idx(N_data,dev_size,random_state)
    ds_train, y_train = df.iloc[train_idx,0].reset_index(drop=True), df.iloc[train_idx,1].reset_index(drop=True)
    ds_dev, y_dev = df.iloc[dev_idx,0].reset_index(drop=True), df.iloc[dev_idx,1].reset_index(drop=True)
    
    return ds_train, y_train, ds_dev, y_dev
    

def train_dev_validation(model,df,random_state=0,
        metrics='accuracy',dev_size=0.1,compute_train=False,**kwargs):
    
    # Divido el dataframe en train y dev
    #tic = time.time()
    ds_train, y_train, ds_dev, y_dev = train_dev_split(df,random_state,dev_size)
    #toc = time.time()
    #print('Splitting time:',toc-tic)

    # Entrenamos el modelo sobre los datos de train
    #tic = time.time()
    model.train(ds_train, y_train,**kwargs)
    #toc = time.time()
    #print('Training time:',toc-tic)

    # Predecimos las nuevas muestras y medimos la performance
    #tic = time.time()
    y_predict = model.predict(ds_dev)
    #toc = time.time()
    #print('Predicting time:',toc-tic)

    #tic = time.time()
    score = get_score(y_dev,y_predict,metrics)
    #toc = time.time()
    #print('Scoring time:',toc-tic)

    #tic = time.time()
    if compute_train:
        dev_score = score
        y_predict = model.predict(ds_train)    
        train_score = get_score(y_train,y_predict,metrics)
        score = {}
        if isinstance(metrics,str):
            metrics = [metrics]
        for metric in metrics:
            score['train_{}'.format(metric)] = train_score[metric]
            score['validation_{}'.format(metric)] = dev_score[metric]
    #toc = time.time()
    #print('Scoring in train time:',toc-tic)

    return score

def get_kfolds_idx(N,k_folds=5,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    indeces = []
    splitted_arrays = np.array_split(rand_idx,k_folds)
    for i in range(1,k_folds+1):
        train_idx = np.hstack(splitted_arrays[:i-1] + splitted_arrays[i:])
        dev_idx = splitted_arrays[i-1]
        indeces.append((train_idx, dev_idx))

    return indeces


def k_fold_validation(model,df,k_folds=5,
        random_state=0,metrics='accuracy'):

    N_data = len(df)
    indices = get_kfolds_idx(N_data,k_folds,random_state)
    scores = []

    for k, (train_idx, dev_idx) in enumerate(indices):
        
        print('Fold {}'.format(k+1))
        ds_train, y_train = df.iloc[train_idx,0].reset_index(drop=True), df.iloc[train_idx,1].reset_index(drop=True)
        ds_dev, y_dev = df.iloc[dev_idx,0].reset_index(drop=True), df.iloc[dev_idx,1].reset_index(drop=True)

        model.train(ds_train,y_train)
        
        y_pred = model.predict(ds_dev)
        score = get_score(y_dev,y_pred,metrics)
        scores.append(score)

    if isinstance(metrics,str):
        metrics = [metrics]

    final_score = {metric: np.mean([score[metric] for score in scores]) for metric in metrics}
    return final_score


def get_score(y_test,y_predict,metrics):
    if isinstance(metrics,str):
        return {metrics: check_performance(y_test,y_predict,metrics)}

    return {metric:check_performance(y_test,y_predict,metric) for metric in metrics}



def check_performance(y_test,y_predict,metric):

    beta = re.findall(r'f(\d+)_score',metric)
    beta_macro = re.findall(r'f(\d+)_macro',metric)
    beta_micro = re.findall(r'f(\d+)_micro',metric)

    if metric == 'confusion_matrix':
        return confusion_matrix(y_test,y_predict)

    elif metric == 'accuracy':
        return accuracy(y_test,y_predict)

    elif metric == 'balanced_accuracy':
        return balanced_accuracy(y_test,y_predict)

    elif metric == 'precision':
        return precision(y_test,y_predict)

    elif metric == 'recall':
        return recall(y_test,y_predict)

    elif len(beta) > 0:
        return f_beta_score(y_test,y_predict,int(beta[0]))

    elif len(beta_macro) > 0:
        return f_beta_macro(y_test,y_predict,int(beta_macro[0]))

    elif len(beta_micro) > 0:
        return f_beta_micro(y_test,y_predict,int(beta_micro[0]))        

    else:
        raise TypeError('Not supported {} metric'.format(metric))



def confusion_matrix(y_test,y_predict):

    if isinstance(y_test,torch.Tensor):
        y_test = y_test.numpy()
    if isinstance(y_predict,torch.Tensor):
        y_predict = y_predict.numpy()

    classes = np.unique(y_test)
    n_classes = len(classes)
    cm = np.zeros((n_classes,n_classes))
    for i, c in enumerate(classes):
        cm[i,:] = (y_predict[y_test == c].reshape(-1,1) == classes).sum(axis=0)
    return cm


def accuracy(y_test,y_predict):

    cm = confusion_matrix(y_test,y_predict)

    return np.diag(cm).sum() / cm.sum()

def precision(y_test,y_predict):

    cm = confusion_matrix(y_test,y_predict)
    out = np.zeros(cm.shape[0])
    x1 = np.diag(cm)
    x2 = cm.sum(axis=0)

    return np.divide(x1,x2,where=(x2 != 0),out=out)


def recall(y_test,y_predict):

    cm = confusion_matrix(y_test,y_predict)
    out = np.zeros(cm.shape[0])
    x1 = np.diag(cm)
    x2 = cm.sum(axis=1)

    return np.divide(x1,x2,where=(x2 != 0),out=out)


def balanced_accuracy(y_test,y_predict):

    rec = recall(y_test,y_predict)
    return rec.mean()


def f_beta_score(y_test,y_predict,beta):
    
    p = precision(y_test,y_predict)
    r = recall(y_test,y_predict)
    result = np.zeros_like(p)
    x1 = p * r
    x2 = beta**2 * p + r
    np.divide(x1,x2,where=x2!=0,out=result)
    return (1+beta**2) * result


def f_beta_macro(y_test,y_predict,beta):
    return f_beta_score(y_test,y_predict,beta).mean()


def f_beta_weighted(y_test,y_predict,beta):
    weights = np.array([(y_test == c).sum() for c in np.unique(y_test)])
    return f_beta_score(y_test,y_predict,beta).dot(weights)


def f_beta_micro(y_test,y_predict,beta):
    pass