import pandas as pd

def load_data(DATA_PATH,split='train',nclasses=2):
    
    path = DATA_PATH + split + '.csv'
    df = pd.read_csv(path,lineterminator='\n',sep=',',usecols=['review_content','review_title','review_rate'])
    
    if nclasses == 2:
        df = df[df['review_rate'] != 3].reset_index(drop=True)
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 1.
        print('Dataset cargado para 2 clases (malo=0, bueno=1)')
    elif nclasses == 3:
        df.loc[(df['review_rate'] <= 2),['review_rate']] = 0.
        df.loc[(df['review_rate'] >= 4),['review_rate']] = 2.
        df.loc[df['review_rate'] == 3,['review_rate']] = 1.
        print('Dataset cargado para 3 clases (malo=0, medio=1, bueno=2)')
    elif nclasses == 5:
        print('Dataset cargado para 5 clases (muy malo=0, malo=2, medio=3, bueno=4 muy bueno=1)')
    else:
        raise TypeError('nclasses must be either 2, 3 or 5')
        
    print('Num samples per category:')
    print(df['review_rate'].value_counts().sort_index())
    return df