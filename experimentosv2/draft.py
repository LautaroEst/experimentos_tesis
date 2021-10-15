import pandas as pd
import numpy as np

def main():
    ds = pd.Series([
        'Esto es una prueba'.split(' '),
        'para ver si funciona lo que creo'.split(' '),
        'que va a funcionar ahora.'.split(' ')
    ])
    y = np.array([1,2,3])
    #df = pd.concat((ds,pd.Series(y)),keys=['x','y'],axis=1)
    #print(df)
    #df = df.sort_values(by=['x'],key=lambda x: x.str.len(),ascending=False)
    #print(df)
    sequence_batch = ds.copy()
    sent_lenghts = sequence_batch.str.len()
    sorted_idx = sent_lenghts.argsort()[::-1]
    sorted_sequence_batch = sequence_batch.iloc[sorted_idx].reset_index(drop=True)
    sorted_sent_lenghts = sent_lenghts.iloc[sorted_idx].tolist()
    y_pred = np.array([2,1,3])
    print(sequence_batch)
    print(sorted_sequence_batch)
    print(sorted_sent_lenghts)
    resorted_idx = sorted_idx.argsort()
    y_pred = y_pred[resorted_idx]
    print(y_pred)


if __name__ == '__main__':
    main()