from collections import defaultdict
from itertools import chain, tee, islice
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import time

def _get_ngrams(doc, ngram_range=(1,1)):

	for n in range(ngram_range[0],ngram_range[1]+1):
	    tlst = doc
	    while True:
	        a, b = tee(tlst)
	        l = tuple(islice(a, n))
	        if len(l) == n:
	            yield ' '.join(l)
	            next(b)
	            tlst = b
	        else:
	            break


class BONgramsVectorizer(object):

	vocab_base = {'[UNK]': 0}

	def __init__(self,tokenizer=None,vocab=None,max_features=None,ngram_range=(1,1)):
		self._tokenizer, self._regex = self._check_and_init_tokenizer(tokenizer)
		self._vocab_arg = self._check_vocab(vocab)
		self.max_features = self._check_max_features(max_features)
		self.ngram_range = self._check_and_init_ngram_range(ngram_range)


	def tokenize(self,ds):
		if self._regex is None:
			return ds.apply(self._tokenizer)
		return ds.str.split(self._regex)


	def _check_and_init_ngram_range(self,ngram_range):
		if isinstance(ngram_range,tuple):
			if len(ngram_range) ==  2:
				return ngram_range
		elif isinstance(ngram_range,int):
			return (1,ngram_range)
		
		raise TypeError('ngram_range must be int or tuple.')


	def _check_and_init_tokenizer(self,tokenizer):
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


	def _init_vocab(self,vocab):
		if vocab is None:
			vocab = defaultdict(lambda x: x, self.vocab_base)
			vocab.default_factory = vocab.__len__
			fixed_vocab = False

		else:
			inv_vocab_base = {idx: tk for tk, idx in self.vocab_base.items()}
			vocab = {tk: idx for idx, tk in chain(inv_vocab_base.items(),enumerate(vocab,len(self.vocab_base)))}
			fixed_vocab = True

		return vocab, fixed_vocab


	def _check_vocab(self,vocab):
		if vocab is not None:
			try:
				vocab_is_string = pd.Series(vocab).apply(lambda x: isinstance(x,str)).all()
				contains_special_token = any([(tk in self.vocab_base) for tk in vocab])
			except:
				raise TypeError('vocab must be iterable of strings')

			if not (vocab_is_string and not contains_special_token):
				raise TypeError('All items in vocab must be strings')
		
		return vocab


	def _check_max_features(self,max_features):
		try:
			if max_features is None:
				is_valid = True
			else:
				is_valid = max_features > len(self.vocab_base)
		except:
			is_valid = False
			
		if not is_valid:
			raise TypeError('max_features must be an integer greater than zero.')

		return max_features


	def _count_ngrams(self,ds,fixed_vocab):
		num_docs = len(ds)
		tic = time.time()
		#ds = ds.apply(lambda x: list(_get_ngrams(x,self.ngram_range)))
		ds = ds.apply(lambda x: list(_get_ngrams(self._tokenizer(x),self.ngram_range)))
		toc = time.time()
		print('getting ngrams time:',toc-tic)

		tic = time.time()
		ds = ds.explode().map(self.vocab)
		toc = time.time()
		print('exploding ngrams time:',toc-tic)

		tic = time.time()
		unk_idx = self.vocab_base['[UNK]']
		if fixed_vocab:
			ds = ds.fillna(unk_idx)
		else:
			freqs = ds.value_counts()
			print(freqs,len(freqs))
			if self.max_features is not None:
				freqs = freqs.iloc[:self.max_features]
			ds[~ds.isin(freqs.index)] = unk_idx
		toc = time.time()
		print('Editing tokens time:',toc-tic)

		tic = time.time()
		num_features = len(self.vocab)
		ds = ds.astype(int)
		ds = ds.groupby(level=0).value_counts()
		toc = time.time()
		print('Counting ngrams time:',toc-tic)

		tic = time.time()
		indices = ds.index.droplevel(0).values
		indptr = np.insert(ds.index.get_level_values(0).value_counts().sort_index().cumsum().values,0,0)
		data = ds.values
		X = csr_matrix((data,indices,indptr),shape=(num_docs,num_features))
		toc = time.time()
		print('Building matrix time:',toc-tic)
		return X


	def fit_transform(self,ds):
		tic = time.time()
		self.vocab, fixed_vocab = self._init_vocab(self._vocab_arg)
		toc = time.time()
		print('Initializing vocab time:',toc-tic)
		tic = time.time()
		#ds = self.tokenize(ds)
		toc = time.time()
		print('tokenizing time:',toc-tic)
		X = self._count_ngrams(ds,fixed_vocab)
		self.vocab = dict(self.vocab)
		return X


	def transform(self,ds):
		ds = self.tokenize(ds)
		X = self._count_ngrams(ds,fixed_vocab=True)
		return X


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
	
	vec = BONgramsVectorizer(tokenizer=None,
							vocab=None,
							max_features=None,
							ngram_range=(1,1))
	
	print(vec.fit_transform(ds).toarray())
	print(vec.transofrm(ds))






if __name__ == '__main__':
	test()

