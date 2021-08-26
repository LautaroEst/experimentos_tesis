from .vectorizers import BONgramsVectorizer, WordByCategoryCBOWVectorizer, Word2IndexVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors

from .vectorizers.reweighting import _get_reweight_fn



class BOWNaiveBayesClassifier(object):

	def __init__(self,alpha=1.0,reweighting=None,*args,**kwargs):
		self.vec = CountVectorizer(*args,**kwargs)
		self.reweight_fn = _get_reweight_fn(reweighting)
		self.clf = MultinomialNB(alpha=alpha)

	def train(self,ds_train,y_train):
		X_train = self.vec.fit_transform(ds_train)
		X_train = self.reweight_fn(X_train.T).T
		self.clf.fit(X_train,y_train)

	def predict(self,ds_val):
		X_val = self.vec.transform(ds_val)
		y_pred = self.clf.predict(X_val)
		return y_pred


class CBOWSoftmaxClassifier(object):

	def __init__(self,reweighting=None,*args,**kwargs):
		self.vec = WordByCategoryCBOWVectorizer(*args,**kwargs)
		self.reweight_fn = _get_reweight_fn(reweighting)
		self.clf = LogisticRegression()

	def train(self,ds_train,y_train):
		X_train = self.vec.fit_transform(ds_train,y_train)
		self.clf.fit(X_train,y_train)

	def predict(self,ds_test):
		X_test = self.vec.transform(ds_test)
		y_pred = self.clf.predict(X_test)
		return y_pred


""" class CBOWSoftmaxClassifier(object):

	def __init__(self,tokenizer=None, min_count=0., max_count=None, max_features=None, 
			unk_token=None, reweighting=None,*args,**kwargs):

		self.vec = WordByCategoryCBOWVectorizer(tokenizer,min_count,max_count,
				max_features,unk_token,reweighting)
		self.clf = LogisticRegression(*args,**kwargs)

	def train(self,ds_train,y_train):
		self.vec.train(ds_train,y_train)	
		X_train = self.vec.vectorize_corpus(ds_train)
		self.clf.fit(X_train,y_train)
	
	def predict(self,ds_test):
		X_test = self.vec.vectorize_corpus(ds_test)
		y_pred = self.clf.predict(X_test)
		return y_pred """

class RecurrentModel(nn.Module):

	def __init__(self,embedding_dim,hidden_size,vocab_size):
		super().__init__()
		self.emb = nn.Embedding(vocab_size+2,embedding_dim,padding_idx=vocab_size+1)
		self.rnn = nn.RNN(embedding_dim,hidden_size=hidden_size,batch_first=True)
		self.linear = nn.Linear(self.rnn.hidden_size,1)

	def forward(self,x):
		x = self.emb(x)
		out, _ = self.rnn(x)
		out = out[:,-1,:].unsqueeze(1)
		out = self.linear(out)
		logits = out
		return logits


class RecurrentSoftmaxClassifier(object):

	def __init__(self,hidden_dim,*args,**kwargs):
		self.vec = self.Word2IndexVectorizer(*args,**kwargs)
		self.hidden_dim = hidden_dim
		print('Loading pretrained vectors...')
		self.pretrained_vectors = KeyedVectors.load('/home/lestien/Documents/Trabajos 2021/melisa/experimentos/02 bias/glove-sbwc.i25.model')
		self.embedding_dim = self.pretrained_vectors.vector_size
		print('Done')
		self.first_time = True
	

	class TextData(Dataset):

		def __init__(self,padded_seqs,y):
			self.padded_seqs = padded_seqs.type(torch.long)
			self.y = torch.from_numpy(y).type(torch.long)

		def __getitem__(self,idx):
			return self.padded_seqs[idx], self.y[idx]

		def __len__(self):
			return self.y.size(0)


	class Word2IndexVectorizer(object):

		def __init__(self,*args,**kwargs):
			self.vec = CountVectorizer(*args,**kwargs)
			self.tokenize = self.vec.build_analyzer()

		def fit_transform(self,ds_train):
			self.vec.fit(ds_train)
			vocab = defaultdict(lambda : -1,self.vec.vocabulary_)
			vocab.default_factory = vocab.__len__
			self.vocab = vocab
			ds_train = ds_train.apply(self.tokenize)
			sorted_idx = ds_train.str.len().argsort()
			idx_seqs = np.array([torch.tensor([vocab[tk] for tk in text]) for text in ds_train],dtype=object)
			idx_seqs = idx_seqs[sorted_idx][::-1]
			packed_seqs = pack_sequence(idx_seqs)
			padded_seqs, lenghts = pad_packed_sequence(packed_seqs,batch_first=True,padding_value=len(vocab)+1)
			return padded_seqs

		def transform(self,ds_test):
			vocab = self.vocab
			ds_test = ds_test.apply(self.tokenize)
			sorted_idx = ds_test.str.len().argsort()
			idx_seqs = np.array([torch.tensor([vocab[tk] for tk in text]) for text in ds_test],dtype=object)
			idx_seqs = idx_seqs[sorted_idx][::-1]
			packed_seqs = pack_sequence(idx_seqs)
			padded_seqs, lenghts = pad_packed_sequence(packed_seqs,batch_first=True)
			return padded_seqs
			
	def fill_embedding_matrix(self,emb,vocab,pretrained_model):
		with torch.no_grad():
			for tk, idx in vocab.items():
				try:
					emb.weight[idx,:].copy_(torch.from_numpy(pretrained_model[tk]))
				except KeyError:
					pass

		return emb

	def resume_training(self,ds_train,y_train,batch_size,device,epochs,learning_rate):
		print('Resuming training...')
		padded_seqs = self.vec.transform(ds_train)
		dataset = self.TextData(padded_seqs,y_train.values)
		dataloader = DataLoader(dataset,shuffle=True,batch_size=batch_size)
		model = self.model

		loss_fn = nn.BCEWithLogitsLoss()
		device = torch.device(device)
		model.to(device)

		model.train()
		optimizer = optim.Adam(model.parameters(),learning_rate)
		try:
			for e in tqdm(range(epochs),ascii=True):
				for i, (x,y) in tqdm(enumerate(dataloader)):
					x = x.to(device=device,dtype=torch.long)
					y = y.to(device=device,dtype=torch.float)

					# Forward pass:
					out = model(x).squeeze()

					# Backward pass:
					optimizer.zero_grad()
					loss = loss_fn(out,y)
					loss.backward()
					
					# Optimizer step:
					optimizer.step()

		except KeyboardInterrupt:
			pass

		self.model = model


	def train(self,ds_train,y_train,batch_size,device,epochs,learning_rate):
		if not self.first_time:
			return self.resume_training(ds_train,y_train,batch_size,device,epochs,learning_rate)
		
		self.first_time = False
		print('Vectorizing data...')
		padded_seqs = self.vec.fit_transform(ds_train)
		dataset = self.TextData(padded_seqs,y_train.values)
		dataloader = DataLoader(dataset,shuffle=True,batch_size=batch_size)

		print('Initializing the model...')
		model = RecurrentModel(self.embedding_dim,self.hidden_dim,len(self.vec.vocab))
		model.emb = self.fill_embedding_matrix(model.emb,self.vec.vocab,self.pretrained_vectors)

		loss_fn = nn.BCEWithLogitsLoss()
		device = torch.device(device)
		model.to(device)

		print('Starting training...')
		model.train()
		optimizer = optim.Adam(model.parameters(),learning_rate)
		try:
			for e in tqdm(range(epochs),ascii=True):
				for i, (x,y) in tqdm(enumerate(dataloader)):
					x = x.to(device=device,dtype=torch.long)
					y = y.to(device=device,dtype=torch.float)

					# Forward pass:
					out = model(x).squeeze()

					# Backward pass:
					optimizer.zero_grad()
					loss = loss_fn(out,y)
					loss.backward()
					
					# Optimizer step:
					optimizer.step()

		except KeyboardInterrupt:
			pass

		self.model = model


	def predict(self,ds_val):
		X_val = self.vec.transform(ds_val)
		y_val = np.zeros(X_val.shape[0])
		self.clf.eval()
		dataset = self.TextData(X_val,y_val)
		device = next(self.clf.parameters()).device
		with torch.no_grad():
			N = len(dataset)
			y_pred = np.zeros(N)
			for i in range(N):
				x, _ = dataset[i]
				x = x.to(device=device,dtype=torch.float)
				out = self.clf(x)
				y_pred[i] = 0 if out <= 0 else 1
			
		return y_pred

