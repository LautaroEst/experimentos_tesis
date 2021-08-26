import pandas as pd


class Melisa2Dataset(object):

	def __init__(self):
		self.data_path = '/home/lestien/Documents/Trabajos 2021/melisa/data/'

	def get_train_dataframe(self,usecols=None):
		df_train = pd.read_csv(self.data_path + 'esp/train.csv',
							lineterminator='\n',
							sep=',',
							usecols=usecols)
		df_train = df_train[df_train['review_rate'] != 3]
		df_train.loc[(df_train['review_rate'] <= 2),['review_rate']] = 0.
		df_train.loc[(df_train['review_rate'] >= 4),['review_rate']] = 1.
		return df_train

	def get_test_dataframe(self,usecols=None):
		df_test = pd.read_csv(self.data_path + 'esp/test.csv',
                    lineterminator='\n',
                    sep=',',
					usecols=usecols)
		df_test = df_test[df_test['review_rate'] != 3]
		df_test.loc[(df_test['review_rate'] <= 2),['review_rate']] = 0.
		df_test.loc[(df_test['review_rate'] >= 4),['review_rate']] = 1.
		return df_test
