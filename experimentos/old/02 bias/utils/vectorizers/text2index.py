import pandas as pd
from collections import defaultdict
from itertools import chain
import numpy as np
#import tensorflow as tf
import torch


class SequenceVectorizer(object):

	special_tokens = {
		'[UNK]': 0,
		'[PAD]': 1,
		'[BOS]': 2,
		'[EOS]': 3,
		'[MSK]': 4,
	}

	def __init__(self,tokenizer=None,vocab=None,max_features=None,max_length_sequence=None,array_type='numpy'):
		self._tokenizer, self._regex = self._check_tokenizer(tokenizer)
		self.vocab, self._fixed_vocab = self._check_vocab(vocab)
		self.max_features = self._check_max_features(max_features)
		self.max_length_sequence = self._check_max_length_sequence(max_length_sequence)

		if array_type not in ['numpy', 'tf', 'torch']:
			raise TypeError('type must be a string of value "numpy", "tf" or "torch".')
		self.type = array_type


	def tokenize(self,ds):
		if self._regex is None:
			return ds.apply(self._tokenizer)
		return ds.str.split(self._regex)


	def _check_max_length_sequence(self,l):
		try:
			if l is None:
				is_valid = True
			else:
				is_valid = l > 0
		except:
			is_valid = False
		if not is_valid:
			raise TypeError('max_length_sequence must be an integer greater than zero.')

		return l


	def _tk2idx(self,ds):
		ds = ds.explode().map(self.vocab)
		unk_idx = self.special_tokens['[UNK]']
		if self._fixed_vocab:
			ds = ds.fillna(unk_idx)
		else:
			freqs = ds.value_counts()
			if self.max_features is not None:
				freqs = freqs[:self.max_features]
			ds[~ds.isin(freqs.index)] = unk_idx

		ds = ds.astype(int)
		ds = ds.groupby(level=0).agg(list)
		return ds


	def _check_tokenizer(self,tokenizer):
		if tokenizer is None:
			tokenizer = ' '
		if callable(tokenizer):
			regex = None
		elif isinstance(tokenizer,str):
			regex = tokenizer
			tokenizer = None
		else:
			raise TypeError('tokenizer must be callable or regex')
		return tokenizer, regex


	def _check_vocab(self,vocab):
		if vocab is None:
			vocab = defaultdict(lambda x: x,self.special_tokens)
			vocab.default_factory = vocab.__len__
			fixed_vocab = False

		else:
			try:
				vocab_is_string = pd.Series(vocab).apply(lambda x: isinstance(x,str)).all()
				contains_special_token = any([(tk in self.special_tokens) for tk in vocab])
			except:
				raise TypeError('vocab must be iterable of strings')
			
			if vocab_is_string and not contains_special_token:
				inv_special_tokens = {idx: tk for tk, idx in self.special_tokens.items()}
				vocab = {tk: idx for idx, tk in chain(inv_special_tokens.items(),enumerate(vocab,len(self.special_tokens)))}
				fixed_vocab = True
			else:
				raise TypeError('All items in vocab must be strings')

		return vocab, fixed_vocab


	def _check_max_features(self,max_features):
		try:
			if max_features is None:
				is_valid = True
			else:
				is_valid = max_features > len(self.special_tokens)
		except:
			is_valid = False
			
		if not is_valid:
			raise TypeError('max_features must be an integer greater than zero.')

		return max_features


	def _pad_sequences(self,ds,sequence_length,padding_value):

		lengths = ds.str.len()
		row_ends = lengths.cumsum()
		pad_lengths = sequence_length - lengths
		where_to_pad = np.repeat(row_ends,pad_lengths)
		values = ds.explode().values
		padded_values = np.insert(values,where_to_pad,padding_value).reshape(-1,sequence_length)
		return padded_values


	def fit_transform(self,ds):

		ds = self.tokenize(ds)
		ds = self._tk2idx(ds)
		self.vocab = dict(self.vocab)

		max_len = ds.str.len().max()
		if self.max_length_sequence is not None:
			if  max_len > self.max_length_sequence:
				raise TypeError('max_length_sequence too small. Some sentences are too large for this value.')
			padded_sequence_length = self.max_length_sequence
		else:
			padded_sequence_length = max_len			
			
		padded_sequences = self._pad_sequences(ds,padded_sequence_length,self.special_tokens['[PAD]'])
		padded_sequences = padded_sequences.astype(np.int64)
		if self.type == 'numpy':
			return padded_sequences
		elif self.type == 'tf':
			return tf.convert_to_tensor(padded_sequences)
		elif self.type == 'torch':
			return torch.from_numpy(padded_sequences).type(torch.long)
		

	def transform(self,ds):
		ds = self.tokenize(ds)
		ds = ds.explode().map(self.vocab)
		unk_idx = self.special_tokens['[UNK]']
		ds = ds.fillna(unk_idx).astype(int).groupby(level=0).agg(list)

		max_len = ds.str.len().max()
		if self.max_length_sequence is not None:
			if  max_len > self.max_length_sequence:
				raise TypeError('max_length_sequence too small. Some sentences are too large for this value.')
			padded_sequence_length = self.max_length_sequence
		else:
			padded_sequence_length = max_len		

		padded_sequences = self._pad_sequences(ds,padded_sequence_length,self.special_tokens['[PAD]'])
		padded_sequences = padded_sequences.astype(np.int64)
		if self.type == 'numpy':
			return padded_sequences
		elif self.type == 'tf':
			return tf.convert_to_tensor(padded_sequences)
		elif self.type == 'torch':
			return torch.from_numpy(padded_sequences).type(torch.long)

		

def test():
	ds = pd.Series(['Las bandas están muy bien armadas, con buen material (cintas, anclajes y arneses) y los agarres son cómodos. La he usado bastante desde que la compre y demuestra resistencia, estoy conforme con el producto. Calidad/precio muy buena, recomendable.',
					'Todo excelente, buena calidad y excelente rendimiento del toner. Lo mejor de todo es que cuentan con garantía.',
					'Debe ser uno de los peores productos que e recibido, cuenta con múltiples fallos de fabrica y no me supieron dar una solución a esto. Me llevo un gran disgusto y unos borcegos inservibles.',
					'Pésimo producto. Funciona. A media marcha las manecillas son muy pesados para un maquinaria tan de pésima calidad.',
					'Barbaro!! mi gato se acostumbro enseguida a pesar de que tiene ya casi un año. Y venia usando bandeja simple. Un exito, en cuanto a higiene y privacidad para el gato, ademas de que no andan mas sueltas las piedritas sanitarias!.',
					'No traía el metraje indicado y una pequeña maraña o deshilado entre los 80 y 100 mts.',
					'Sinceramente lo compre con desconfianza pero es excelente. Realmente saca los olores y me permitió no tener que tomar pastillas para la alergia. Al principio se siente mucho un olor como “cloro” pero luego ya no se percibe. En relación al ruido es mínimo. El alcance es muy bueno ( llega a cubrir 2 ambientes sin problemas).',
					'Me gustó mucho se adapto a mi necesidad justo lo quería , el material es bueno y se siente bien al ponérsela ya que tiene otra tela distinta por dentro la recomiendo.',
					'Muy buen producto! no le doy 5 estrellas ya que con el uso me mostró una muy pequeña fuga de aire que no altera el rendimiento de la bomba.',
					'Muy buena, ligera, y le encanto al usuario. Si quieres algo resistente, económico y además de una buena marca como lo es prinsel es tu mejor opción. El único detalle que yo note es que no es muy alta, si tu bebe es muy grande no la recomiendo.'])
	
	vec = SequenceVectorizer(tokenizer=None,
							vocab=None,
							max_features=None,
							max_length_sequence=None)
	
	print(vec.fit_transform(ds,'tf'))




if __name__ == '__main__':
	test()


