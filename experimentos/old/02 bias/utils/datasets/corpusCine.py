import os
DATASET_PATH = '/home/lestien/Documents/Trabajos 2021/melisa/other_datasets/corpusCine/corpusCriticasCine'

import xml.etree.ElementTree as ET
import pandas as pd
import glob


def reader():
	
	xml_files = glob.glob('{}/*.xml'.format(DATASET_PATH))
	for filename in xml_files:
		print(filename)
		with open(filename,'r',encoding='utf-8') as f:
			xml_file = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n{}'.format(f.read())
		review = ET.fromstring(xml_file)
		#review = tree.getroot()
		summary = review[0].text
		text = review[1].text
		label = review.attrib['rank']
		yield text, summary, label



class CorpusCineDataset(object):

	def __init__(self):
		pass

	def get_dataframe(self):
		reviews, summaries, labels = zip(*[(text,summary,label) for text, summary, label in reader()])
		return pd.DataFrame({'review':reviews, 'summary': summaries, 'label':labels})


	