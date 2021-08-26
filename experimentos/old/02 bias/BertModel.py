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
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification


class BERTClassifier(object):
	"""
	Implementación de un clasificador con BERT for Classification
	"""

	def __init__(self,beto_source='bert/',max_len=512):
		self.beto_source = beto_source
		self.max_len = max_len
		self.criterion = torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, 
										reduce=None, reduction='mean', pos_weight=None)

	def vectorize_sentences(self,ds,beto_source,max_len):
		tokenizer = BertTokenizer.from_pretrained(beto_source, do_lower_case=True)
		#indices = [[tokenizer.convert_tokens_to_ids(tk) for tk in tokenizer.tokenize("[CLS] {} [SEP]".format(sentence))] for sentence in ds]
		""" attention_masks = []
		indices = []
		for sentence in ds:
			tokens = tokenizer.tokenize("[CLS] {} [SEP]".format(sentence))
			ids = [tokenizer.convert_tokens_to_ids(tk) for tk in tokens]
			ids_len = len(ids)
			ids.extend(np.zeros(max_len-ids_len,dtype=int))
			att_idx = np.zeros((max_len,),dtype=float)
			att_idx[:ids_len] = 1.
			indices.append(ids)
			attention_masks.append(list(att_idx))
		return indices, attention_masks """
		numdocs = len(ds)
		attention_masks = np.zeros((numdocs,max_len),dtype=float)
		indices = np.zeros((numdocs,max_len),dtype=int)
		for i,sent in enumerate(ds):
			tokens = tokenizer.tokenize("[CLS] {} [SEP]".format(sent))
			numtokens = len(tokens)
			indices[i,:numtokens] = [tokenizer.convert_tokens_to_ids(tk) for tk in tokens]
			attention_masks[i,:numtokens] = 1.
		return indices, attention_masks

	def train(self,ds_train,y_train,batch_size=32,epochs=1,learning_rate=1e-3,device='cpu'):

		print('Vectorizing sentences...')
		indices, attention_masks = self.vectorize_sentences(ds_train,self.beto_source,self.max_len)
		indices = torch.from_numpy(indices).type(torch.long)
		attention_masks = torch.from_numpy(attention_masks).type(torch.float)
		labels = torch.from_numpy(y_train.values).type(torch.float).view(-1,1)
		dataset = TensorDataset(indices,attention_masks,labels)
		dataloader = DataLoader(dataset,shuffle=True,batch_size=batch_size)

		print('Initializing the model...')
		self.device = torch.device(device)
		config = BertConfig(vocab_size=31002, # Tamaño del vocabulario
                    hidden_size=768, # Dimensión de los word embeddings
                    num_hidden_layers=12, # Cantidad de capas del transformer
                    num_attention_heads=12, # Cantidad de cabezas de atención por capa
                    intermediate_size=3072, # Dimensión intermedia de las capas lineales
                    hidden_act='gelu',  # Activación intermedia entre las capas lineales
                    hidden_dropout_prob=0.1, # Dropout entre las capas lineales
                    attention_probs_dropout_prob=0.1, # Dropout de las capas de atención
                    num_labels=1) # Cantidad de categorías a la salida

		self.model = BertForSequenceClassification.from_pretrained(self.beto_source, config=config)
		self.model.to(self.device)
		
		print('Starting training...')
		model = self.model
		model.train()
		criterion = self.criterion
		optimizer = optim.Adam(model.parameters(),lr=learning_rate)
		loss_history = []
		print(model)

		try:
			for e in range(epochs):
				for i, batch in enumerate(dataloader):
					b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

					optimizer.zero_grad()
					out = model(b_input_ids, token_type_ids=None, 
								attention_mask=b_input_mask, labels=b_labels)
					logits = out.logits
					loss = criterion(logits,b_labels)
					loss.backward()
					optimizer.step()
					loss_history.append(loss.item())

					#if i % 50 == 49:
					print('Epoch: {}. Loss value: {}'.format(e,loss.item()))
					#print([torch.norm(param.grad).item() for param in model.parameters()])

				#self.check_accuracy(model,dataset)

		except KeyboardInterrupt:
			print('Interrupting training')
			print('Loss value: {}'.format(loss.item()))

		self.model = model
		self.loss_history = loss_history
		self.optimizer = optimizer
		self.epoch = e


	def predict(self,ds_test):
		indices, attention_masks = self.vectorize_sentences(ds_test,self.beto_source,self.max_len)
		indices = torch.from_numpy(indices).type(torch.long)
		attention_masks = torch.from_numpy(attention_masks).type(torch.float)
		dataset = TensorDataset(indices,attention_masks)

		model = self.model
		model.eval()

		y_pred = []
		for i in range(len(dataset)):
			ind, att = dataset[i]
			ind = ind.unsqueeze(0).to(device=self.device,dtype=torch.long)
			att = att.unsqueeze(0).to(device=self.device,dtype=torch.long)
			out = model(ind, token_type_ids=None, attention_mask=att)
			logits = out.logits
			y_pred.append(1 if logits > 0 else 0)
		
		return np.array(y_pred)


""" # %%
import pandas as pd
from transformers import BertTokenizer
import numpy as np
import re

ds_train = pd.Series(['Esto es un corpus de prueba', 'para saber si esto es un prueba', 'si anda bien, bien', 'si no, no deberia andar bien'])
y_train = np.array([1,0,0,1])
ds_test = pd.Series(['Esto es el test', 'es un corpus distinto a train'])
y_test = np.array([1,0])
token_pattern = re.compile(r'[\w]+|[!¡¿\?\.,\'"]')

# Tokenización:
def vectorize_sentences(ds,beto_source,max_len):
	tokenizer = BertTokenizer.from_pretrained(beto_source, do_lower_case=True)
	#indices = [[tokenizer.convert_tokens_to_ids(tk) for tk in tokenizer.tokenize("[CLS] {} [SEP]".format(sentence))] for sentence in ds]
	attention_masks = []
	indices = []
	for sentence in ds:
		tokens = tokenizer.tokenize("[CLS] {} [SEP]".format(sentence))
		ids = [tokenizer.convert_tokens_to_ids(tk) for tk in tokens]
		ids_len = len(ids)
		ids.extend(np.zeros(max_len-ids_len,dtype=int))
		att_idx = np.zeros((max_len,),dtype=float)
		att_idx[:ids_len] = 1.
		indices.append(ids)
		attention_masks.append(list(att_idx))
	return indices, attention_masks

beto_source = 'bert/'
max_len = 20
indices, att_masks = vectorize_sentences(ds_train,beto_source,max_len)

print(indices)
print(att_masks)
#clf = RecurrentClassifier(tokenizer=lambda x: token_pattern.findall(x))
#clf.train(ds_train,y_train,batch_size=2,epochs=10,learning_rate=1e-3,device='cpu') """


def validation():
	random_seed = 12738
	df_all = Melisa2Dataset().get_train_dataframe(usecols=['review_content','review_rate']).reset_index(drop=True).sample(n=10000,random_state=random_seed).reset_index(drop=True)

	""" ds_train = pd.Series(['Esto es un corpus de prueba', 'para saber si esto es un prueba', 'si anda bien, bien', 'si no, no deberia andar bien'])
	y_train = np.array([1,0,0,1])
	ds_test = pd.Series(['Esto es el test', 'es un corpus distinto a train'])
	y_test = np.array([1,0]) """

	max_features = 10000
	token_pattern = re.compile(r'[\w]+|[!¡¿\?\.,\'"]')
	tokenizer = lambda x: token_pattern.findall(x)
	clf = BERTClassifier(beto_source='/home/lestien/Documents/Trabajos 2021/melisa/experimentos/02 bias/bert/',max_len=512)
	""" clf.train(ds_train,y_train,batch_size=32,epochs=1,
			learning_rate=1e-5,device='cuda:1') """
	""" clf.train(df_all['review_content'],df_all['review_rate'].values,batch_size=128,epochs=600,
			learning_rate=1e-3,device='cuda:1') """
	score = train_dev_validation(clf,df_all,random_state=random_seed,metrics='accuracy',
			dev_size=0.05,compute_train=False,batch_size=8,epochs=1,
			learning_rate=1e-2,device='cuda:1')

	print('Accuracy: {:.2f}%'.format(score['accuracy']*100))
	torch.save({
		'epoch': clf.epoch,
		'model': clf.model,
		'optimizer': clf.optimizer,
		'loss_history': clf.loss_history
	},'03-06-2021-0000_model.pkl')
	

def test():
	pass


if __name__ == '__main__':
	validation()

