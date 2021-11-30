import os
import xml.etree.ElementTree as ET
import pandas as pd

DATASET_PATH = os.path.join(os.getcwd(),'../../other_datasets/tass2012')

def read_tass(split='train', nclasses=2):
    tree = ET.parse(os.path.join(DATASET_PATH,'general-' + split + '-tagged.xml'))
    root = tree.getroot()
    if nclasses == 2:
        label2num = {'P+': 1, 'P': 1, 'N': 0, 'N+': 0}
    elif nclasses == 3:
        label2num = {'P+': 2, 'P': 2, 'NEU': 1, 'N': 0, 'N+': 0}
    elif nclasses == 5:
        label2num = {'P+': 4, 'P': 3, 'NEU': 2, 'N': 1, 'N+': 0}
    
    dataset = {'tweet': [], 'label': []}
    for item in root:
        tweet = item[2].text
        label = item[5][0][0].text
        if (label == 'NONE') or (label == 'NEU' and nclasses == 2):
            continue
        num = label2num[label]
        dataset['tweet'].append(tweet)
        dataset['label'].append(num)

    dataset = pd.DataFrame.from_dict(dataset)
    return dataset
        


def main():
    df = read_tass(split='test', nclasses=3)
    print(df)
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()