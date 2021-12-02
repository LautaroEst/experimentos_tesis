import xml.etree.ElementTree as ET
import pandas as pd
import os

RANDOM_SEED = 61273812
SEMEVAL_PATH = os.path.join(os.getcwd(),'../../other_datasets/semeval2016_task5_sbt1')


def load_semeval(split='train'):
    if split == 'train':
        path = os.path.join(SEMEVAL_PATH,"SemEval-2016ABSA Restaurants-Spanish_Train_Subtask1.xml")
    elif split == 'test':
        path = os.path.join(SEMEVAL_PATH,"SP_REST_SB1_TEST.xml.gold")

    aspect2idx = {
        'RESTAURANT': 0, 'FOOD': 1, 'DRINKS': 2, 
        'AMBIENCE': 3, 'SERVICE': 4, 'LOCATION': 5
    }
    polarity2idx = {'positive': 1, 'negative': 2, 'neutral': 3, 'conflict': 4}

    tree = ET.parse(path)
    root = tree.getroot()
    samples = {'sentence': [], 'polarity': [], 'aspect': []}
    for i, review in enumerate(root):
        for sentence in review[0]:
            sent_text = sentence[0].text
            samples['sentence'].append(sent_text)
            sent_pol = [0] * len(sent_text)
            sent_aspect = [0] * len(sent_text)
            for opinion in sentence[1]:
                if opinion.get('target') == "NULL":
                    continue
                aspect, _ = opinion.get('category').split('#')
                pol = opinion.get('polarity')
                f, t = int(opinion.get('from')), int(opinion.get('to'))
                pol_idx = polarity2idx[pol]
                aspect_idx = aspect2idx[aspect]
                for i in range(f,t,1):
                    sent_pol[i] = pol
                    sent_aspect[i] = aspect
            samples['polarity'].append(sent_pol)
            samples['aspect'].append(sent_aspect)
    # print(samples['sentence'][1])
    # print(samples['polarity'][1])
    # print(samples['aspect'][1])
    # print(len(samples['sentence']))
    df = pd.DataFrame.from_dict(samples)
    return df
