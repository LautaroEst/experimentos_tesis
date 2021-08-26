import os
DATASET_PATH = '/home/lestien/Documents/Trabajos 2021/melisa/other_datasets/ABSA2016/'

import xml.etree.ElementTree as ET
import pandas as pd


def reader(split='train'):
	if split == 'train':
		filename = 'SemEval-2016ABSA Restaurants-Spanish_Train_Subtask1.xml'
	elif split == 'test_gold':
		filename = 'SP_REST_SB1_TEST.xml.gold'
	tree = ET.parse(os.path.join(DATASET_PATH,filename))
	reviews = tree.getroot()
	for review in reviews:
		sentences = review[0]
		for sentence in sentences:
			text = sentence[0].text
			opinions = [opinion.attrib['polarity'] for opinion in sentence[1]]
			if len(opinions) > 0:
				opinion_whole = max(set(opinions), key=opinions.count)
				yield text, opinion_whole


class ABSA2016PolarityDataset(object):

	def __init__(self):
		pass

	def get_train_dataframe(self):
		reviews, labels = zip(*[(text,label) for text, label in reader('train')])
		return pd.DataFrame({'review':reviews, 'label':labels})

	def get_test_gold_dataframe(self):
		reviews, labels = zip(*[(text,label) for text, label in reader('test_gold')])
		return pd.DataFrame({'review':reviews, 'label':labels})
