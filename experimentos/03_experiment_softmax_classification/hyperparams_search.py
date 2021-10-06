from numpy.core.arrayprint import _leading_trailing
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import sys
sys.path.append('/home/lestien/Documents/Trabajos 2021/melisa/experimentos/')
import utils as ut



DATA_PATH = '/home/lestien/Documents/Trabajos 2021/melisa/datav2/esp/'

def normalize_dataset(df):
    # Pasamos a minúscula todo
    df['review_content'] = df['review_content'].str.lower()
    # Sacamos todos los acentos
    for rep, rep_with in [('[óòÓöøôõ]','o'), ('[áàÁäåâãÄ]','a'), ('[íìÍïîÏ]','i'), 
                            ('[éèÉëêÈ]','e'), ('[úüÚùûÜ]','u'), ('[ç¢Ç]','c'), 
                            ('[ý¥]','y'),('š','s'),('ß','b'),('\x08','')]:
        df['review_content']  = df['review_content'].str.replace(rep,rep_with,regex=True)
    return df

class CatBOWVectorizer(object):

        def __init__(self,token_pattern,max_features,labels_names):
                self.vec = TfidfVectorizer(token_pattern=token_pattern,
                        ngram_range=(1,1),max_features=max_features)
                self.labels_names = labels_names
        
        def _vectorizer_by_cat(self,X,labels):
                n_cats = len(self.labels_names)
                word_vecs = np.zeros((X.shape[1],n_cats),dtype=float)
                for i,n in enumerate(self.labels_names):
                        word_vecs[:,i] = X[labels == n,:].sum(axis=0)
                X_cats = X @ word_vecs
                self.learned_words = word_vecs
                return X_cats

        def fit_transform(self,corpus,labels):
                X = self.vec.fit_transform(corpus)
                X_cats = self._vectorizer_by_cat(X,labels)
                self.learned_mean = X_cats.mean(axis=0,keepdims=True)
                X_cats = X_cats - self.learned_mean
                return X_cats
        
        def transform(self,corpus):
                X = self.vec.transform(corpus)
                X_cats = X @ self.learned_words
                X_cats = X_cats - self.learned_mean
                return X_cats

class SoftmaxClassifier(object):

    def __init__(self,hidden_size,n_iters,batch_size,learning_rate,
            weight_decay,n_classes,device='cuda:1'):
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if n_classes == 2:
            self.model = self.TorchModel(n_classes,hidden_size,1)
        else:
            self.model = self.TorchModel(n_classes,hidden_size,n_classes)
        self.n_classes = n_classes
        self.device = device

    class TorchModel(nn.Module):
        def __init__(self,input_features,hidden_features,output_features):
            super().__init__()
            self.linear1 = nn.Linear(input_features,hidden_features,bias=True)
            self.linear2 = nn.Linear(hidden_features,output_features,bias=True)
        
        def forward(self,x):
            x = torch.relu(self.linear1(x))
            log_probs = self.linear2(x)
            return log_probs

    def fit(self,X,y):
        device = torch.device(self.device)
        X = torch.from_numpy(X).type(torch.float)
        if self.n_classes == 2:
            y = torch.from_numpy(y.reshape(-1,1)).type(torch.float)
        else:
            y = torch.from_numpy(y).type(torch.long)
        dataset = TensorDataset(X,y)
        dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        model = self.model
        model.to(device)
        optimizer = optim.Adam(model.parameters(),lr=self.learning_rate,
                            weight_decay=self.weight_decay)

        if self.n_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        model.train()
        loss_history = []
        for e in range(self.n_iters):
            for i, batch in enumerate(dataloader):
                x, y = (t.to(device=device) for t in batch)

                # Forward
                logprobs = model(x)
                loss = criterion(logprobs,y)

                # Backward
                optimizer.zero_grad()
                loss.backward()

                # Update
                optimizer.step()

            loss_history.append(loss.item())

        model.eval()
        self.model = model
        return loss_history

    def predict(self,X):
        model = self.model
        model.eval()
        device = torch.device(self.device)

        X = torch.from_numpy(X).type(torch.float)
        if self.n_classes == 2:
            activation = lambda logprobs: (logprobs > 0).type(torch.int).detach().view(-1).cpu().tolist()
        else:
            activation = lambda logprobs: logprobs.argmax(dim=1).type(torch.int).detach().view(-1).cpu().tolist()
        dataset = TensorDataset(X)
        dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=False)

        predicted_labels = []
        with torch.no_grad():
            for batch in dataloader:
                x, = (x.to(device) for x in batch)
                y_pred = activation(model(x))
                predicted_labels.extend(y_pred)
        
        return np.array(predicted_labels)

def main():
    df = ut.load_data(DATA_PATH,'train',nclasses=5).loc[:,['review_content','review_rate']]
    #df = df.sample(frac=0.01,random_state=27381)

    df = normalize_dataset(df)
    df_train, df_dev = ut.train_dev_split(df,dev_size=0.1,random_state=2376482)
    y_train, y_dev = df_train['review_rate'].values-1, df_dev['review_rate'].values-1

    max_words = [10000,20000,50000,100000]
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    weight_decays = [0., 0.001]
    results = {mw: {lr: {wd: {} for wd in weight_decays} for lr in learning_rates} for mw in max_words}
    for mw in max_words:
        for lr in learning_rates:
            for wd in weight_decays:
                vec = CatBOWVectorizer(token_pattern=r'(\w+|[\.,!\(\)"\-:\?/%;¡\$\'¿\\]|\d+)',
                            max_features=mw,labels_names=[0,1,2,3,4])

                clf = SoftmaxClassifier(hidden_size=4,n_iters=20,batch_size=256,
                        learning_rate=lr,weight_decay=wd,n_classes=5,device='cuda:1')
    
                X_train = vec.fit_transform(df_train['review_content'],y_train)
                loss_history = clf.fit(X_train,y_train)

                X_dev = vec.transform(df_dev['review_content'])
                y_pred = clf.predict(X_dev)

                print('Training: max words {}, learning rate {}, weight decay {}'.format(mw,lr,wd))
                print('Accuracy: {:.2f}%'.format( (y_dev == y_pred).mean()*100 ))
                results[mw][lr][wd]['y'] = y_dev.copy()
                results[mw][lr][wd]['y_pred'] = y_pred.copy()
                results[mw][lr][wd]['loss_history'] = loss_history


if __name__ == '__main__':
    main()