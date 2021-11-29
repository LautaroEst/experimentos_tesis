import os
import pandas as pd
from transformers import BertTokenizer


corpus = [
    'Esto es una prueba',
    'para ver si funciona lo que creo',
    'que va a funcionar ahora.'
]

MELISA_PATH = '/'.join(os.getcwd().split('/')[:-2]) + '/datav2/esp/'

def load_melisa(split='train',nclasses=2):
    
    path = MELISA_PATH + split + '.csv'
    df = pd.read_csv(path,
                     lineterminator='\n',
                     sep=',',
                     usecols=['review_content','review_rate'],
                     dtype={'review_content': str, 'review_rate': int})
    
    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1.
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2.
        df.loc[df['review_rate'] == 3,['review_rate']] = 1.
    else:
        df['review_rate'] = df['review_rate'] - 1

    return df

def main():
    tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
    corpus = load_melisa('train',nclasses=5)['review_content'].iloc[:100].tolist()
    encoded_inputs = tokenizer(corpus)
    print(type(encoded_inputs))
    # for data in encoded_inputs:
    #     print(data)
    encoded_inputs = pd.DataFrame.from_dict({
        'input_ids': encoded_inputs.input_ids,
        'token_type_ids': encoded_inputs.token_type_ids,
        'attention_mask': encoded_inputs.attention_mask
    })
    print(type(encoded_inputs.iloc[4,:]))
    # for i, (ids, ttype, mask) in encoded_inputs.iterrows():
    #     print(ids, ttype, mask)

    

if __name__ == "__main__":
    main()