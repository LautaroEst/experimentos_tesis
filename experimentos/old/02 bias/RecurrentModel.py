from typing import Reversible
from torch import optim
from torch.nn.modules import activation, dropout
from torch.nn.modules.rnn import LSTM
from torch.optim import optimizer
from torch.utils import data
from utils import Melisa2Dataset
import re
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

from utils.evaluation import train_dev_validation

class SequenceVectorizer(object):

	def __init__(self,tokenizer=None,vocab=None,max_features=None,ngram_range=(1,1)):
		if isinstance(max_features,int) and max_features > 2:
			max_features = max_features - 2
		elif max_features is None:
			pass
		else:
			raise TypeError('max_features must be an integer greater than 2')
		self.sklearn_vec = CountVectorizer(tokenizer=tokenizer,vocabulary=vocab,
                                   max_features=max_features,ngram_range=ngram_range,
                                   lowercase=True,token_pattern=None)
		self.tokenizer = self.sklearn_vec.build_analyzer()
	
	def _build_vocab_from_sklearn(self,vec):
		vocab = vec.vocabulary_		
		inv_dict = {idx:tk for tk,idx in vocab.items()}
		zero_tk, one_tk = inv_dict[0], inv_dict[1]
		vocab['[UNK]'] = vocab.pop(zero_tk)
		vocab['[PAD]'] = vocab.pop(one_tk)
		vocab[zero_tk] = len(vocab)
		vocab[one_tk] = len(vocab)
		return vocab

	def _tk2idx(self,ds):
		print(ds)
		ds = ds.apply(self.tokenizer)
		print(ds)
		ds = ds.apply(lambda x: [self.vocab.get(tk,0) for tk in x])
		print(ds)
		""" print(ds)
		ds = ds.explode()
		print(ds)
		ds = ds.map(self.vocab)
		print(ds)
		ds = ds.fillna(0).astype(int)
		print(ds)
		ds = ds.groupby(level=0).agg(list)
		print(ds) """
		return ds

	def _pad_sequences(self,ds,padding_value):

		lengths = ds.str.len()
		row_ends = lengths.cumsum()
		sequence_length = lengths.max()
		pad_lengths = sequence_length - lengths
		where_to_pad = np.repeat(row_ends,pad_lengths)
		values = ds.explode().values
		padded_values = np.insert(values,where_to_pad,padding_value).reshape(-1,sequence_length)
		return padded_values

	def fit_transform(self,ds_train):
		self.sklearn_vec.fit(ds_train)
		self.vocab = self._build_vocab_from_sklearn(self.sklearn_vec)
		ds_train = self._tk2idx(ds_train)
		X_train = self._pad_sequences(ds_train,1).astype(np.int64)
		print(X_train)
		return X_train

	def transform(self,ds_test):
		ds_test = self._tk2idx(ds_test)
		X_test = self._pad_sequences(ds_test,1).astype(np.int64)
		return X_test


class SequenceVectorizer2(object):

	def __init__(self,tokenizer=None,vocab=None,max_features=None,max_len=512):
		if isinstance(max_features,int) and max_features > 2:
			max_features = max_features - 2
		elif max_features is None:
			pass
		else:
			raise TypeError('max_features must be an integer greater than 2')
		self.sklearn_vec = CountVectorizer(tokenizer=tokenizer,vocabulary=vocab,
                                   max_features=max_features,lowercase=True,token_pattern=None,ngram_range=(1,1))
		self.tokenizer = self.sklearn_vec.build_analyzer()
		self.max_len = max_len

	def _build_vocab_from_sklearn(self,vec):
		vocab = vec.vocabulary_		
		inv_dict = {idx:tk for tk,idx in vocab.items()}
		zero_tk, one_tk = inv_dict[0], inv_dict[1]
		vocab['[UNK]'] = vocab.pop(zero_tk)
		vocab['[PAD]'] = vocab.pop(one_tk)
		vocab[zero_tk] = len(vocab)
		vocab[one_tk] = len(vocab)
		return vocab

	def _tk2idx(self,ds):
		indices = []
		tokenizer = self.tokenizer
		convert_tokens_to_ids = self.vocab.get
		max_len = self.max_len
		indices = np.ones((len(ds),max_len),dtype=int)
		for i,sent in enumerate(ds):
			tokens = tokenizer(sent)
			indices[i,:len(tokens)] = [convert_tokens_to_ids(tk,0) for tk in tokens]
		return indices


	def fit_transform(self,ds_train):
		self.sklearn_vec.fit(ds_train)
		self.vocab = self._build_vocab_from_sklearn(self.sklearn_vec)
		X_train = self._tk2idx(ds_train)
		return X_train

	def transform(self,ds_test):
		X_test = self._tk2idx(ds_test)
		return X_test


""" ds_train = pd.Series(['Esto es un corpus de prueba', 'para saber si esto es un prueba', 
					'si anda bien, bien', 'si no, no deberia andar bien'])
y_train = np.array([1,0,0,1])
ds_test = pd.Series(['Esto es el test', 'es un corpus distinto a train'])
y_test = np.array([1,0])
token_pattern = re.compile(r'[\w]+|[!¡¿\?\.,\'"]')
vec = SequenceVectorizer2(tokenizer=lambda x: token_pattern.findall(x),max_len=20)
vec.fit_transform(ds_train)
print(vec.vocab) """


class RecurrentClassifier(object):
	"""
	Implementación de un clasificador con word embeddings + RNN + Softmax
	"""

	def __init__(self,tokenizer=None,vocab=None,max_features=None,embedding_dim=100,
				hidden_size=100,num_layers=1,dropout=0.,max_len=512):
		self.vec = SequenceVectorizer2(tokenizer,vocab,max_features,max_len)
		self.embedding_dim = embedding_dim
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.dropout = dropout
		self.criterion = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, 
										reduce=None, reduction='mean', pos_weight=None)
		self.first_time = True

	class RecurrentModel(nn.Module):

		def __init__(self,num_embeddings,embedding_dim,hidden_size,num_layers,dropout):
			super().__init__()
			self.emb = nn.Embedding(num_embeddings,embedding_dim,1)
			self.rnn = nn.RNN(input_size=embedding_dim,hidden_size=hidden_size,
						num_layers=num_layers, batch_first=True,dropout=dropout,nonlinearity='relu')
			self.linear = nn.Linear(hidden_size,1)
			#self.linear = nn.Linear(embedding_dim,1)


		def forward(self,x):
			out = self.emb(x)
			#_, (h_n,_) = self.rnn(out)
			_, h_n = self.rnn(out)
			out = self.linear(h_n)
			""" out = torch.sum(out,dim=1)
			out = torch.relu(out)
			out = self.linear(out) """
			return out.view(-1)

	class ReviewsDataset(Dataset):

		def __init__(self,X,y):
			self.X = X
			self.y = y
		
		def __getitem__(self,idx):
			X = torch.from_numpy(self.X[idx,:]).astype(torch.long)
			y = torch.from_numpy(self.y[idx]).astype(torch.float)
			return X, y

		def __len__(self):
			return len(self.y)


	def train(self,ds_train,y_train,batch_size=32,epochs=1,learning_rate=1e-3,device='cpu'):

		print('Vectorizing sentences...')
		X_train = self.vec.fit_transform(ds_train)
		X_train = torch.from_numpy(X_train).type(torch.long)
		y_train = torch.from_numpy(y_train.values).type(torch.float)
		dataset = TensorDataset(X_train,y_train)
		#dataset = self.ReviewsDataset(X_train,y_train)
		dataloader = DataLoader(dataset,shuffle=True,batch_size=batch_size)

		print('Initializing the model...')
		self.device = torch.device(device)
		self.model = self.RecurrentModel(len(self.vec.vocab),self.embedding_dim,
							self.hidden_size,self.num_layers,self.dropout)
		self.model.to(self.device)
		self.starting_embeddings = dict(self.model.named_parameters())['emb.weight']
		
		print('Starting training...')
		model = self.model
		model.train()
		criterion = self.criterion
		optimizer = optim.Adam(model.parameters(),lr=learning_rate)
		loss_history = []
		print(model)
		print(['{}: {}'.format(name,torch.norm(param.data).item()) for name,param in model.named_parameters()])
		try:
			for e in range(epochs):
				for i, (x,y) in enumerate(dataloader):
					x = x.to(device=self.device,dtype=torch.long)
					y = y.to(device=self.device,dtype=torch.float)

					logits = model(x)

					optimizer.zero_grad()
					loss = criterion(logits,y)
					loss.backward()

					optimizer.step()
					loss_history.append(loss.item())

					if i % 100 == 0:
						print('Epoch: {}. Loss value: {}'.format(e,loss.item()))
						#print([torch.norm(param.grad).item() for param in model.parameters()])
						print(torch.norm(dict(model.named_parameters())['emb.weight'].grad))
						

				#self.check_accuracy(model,dataset)

		except KeyboardInterrupt:
			print('Interrupting training')
			print('Loss value: {}'.format(loss.item()))

		self.model = model
		self.loss_history = loss_history
		self.optimizer = optimizer
		self.epoch = e
		self.final_embeddings = dict(self.model.named_parameters())['emb.weight']

	""" def _check_accuracy(self,model,dataset):
		model.eval()
		indices = np.
 """
	def predict(self,ds_test):
		X_test = self.vec.transform(ds_test)
		X_test = torch.from_numpy(X_test).type(torch.long)
		#dataset = TensorDataset(X_test)

		model = self.model
		model.eval()

		y_pred = []
		for i in range(X_test.size(0)):
			x = X_test[i,:].unsqueeze(0)
			x = x.to(device=self.device,dtype=torch.long)
			log_prob = model(x)
			y_pred.append(1 if log_prob > 0 else 0)
		
		return np.array(y_pred)
""" 

ds_train = pd.Series(['Esto es un corpus de prueba', 'para saber si esto es un prueba', 'si anda bien, bien', 'si no, no deberia andar bien'])
y_train = np.array([1,0,0,1])
ds_test = pd.Series(['Esto es el test', 'es un corpus distinto a train'])
y_test = np.array([1,0])
token_pattern = re.compile(r'[\w]+|[!¡¿\?\.,\'"]')
clf = RecurrentClassifier(tokenizer=lambda x: token_pattern.findall(x))
clf.train(ds_train,y_train,batch_size=2,epochs=10,learning_rate=1e-3,device='cpu') """


def validation():
	#random_seed = 12738
	df_all = Melisa2Dataset().get_train_dataframe(usecols=['review_content','review_rate']).reset_index(drop=True)#.sample(n=100,random_state=None).reset_index(drop=True)
	""" ds_train = pd.Series(['Esto es un corpus de prueba', 'para saber si esto es un prueba', 'si anda bien, bien', 'si no, no deberia andar bien'])
	y_train = np.array([1,0,0,1])
	ds_test = pd.Series(['Esto es el test', 'es un corpus distinto a train'])
	y_test = np.array([1,0]) """

	max_features = 10000
	token_pattern = re.compile(r'[\w]+|[!¡¿\?\.,\'"]')
	tokenizer = lambda x: token_pattern.findall(x)
	clf = RecurrentClassifier(tokenizer=tokenizer,max_features=max_features,
						embedding_dim=50,hidden_size=100,max_len=512)
	""" clf.train(ds_train,y_train,batch_size=32,epochs=1,
			learning_rate=1e-5,device='cuda:1') """
	""" clf.train(df_all['review_content'],df_all['review_rate'].values,batch_size=128,epochs=600,
			learning_rate=1e-3,device='cuda:1') """
	score = train_dev_validation(clf,df_all,random_state=None,metrics='accuracy',
			dev_size=0.05,compute_train=False,batch_size=128,epochs=10,
			learning_rate=1e-2,device='cuda:1')
	print(clf.starting_embeddings)
	print(clf.final_embeddings)
	#print('Accuracy: {:.2f}%'.format(score['accuracy']*100))
	torch.save({
		'epoch': clf.epoch,
		'model': clf.model,
		'optimizer': clf.optimizer,
		'loss_history': clf.loss_history
	},'02-06-2021-2245_model.pkl')
	

def test():
	pass


if __name__ == '__main__':
	validation()


