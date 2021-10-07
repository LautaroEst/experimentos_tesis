import pandas as pd
import numpy as np

def main():
    ds = pd.Series([
        'Esto es una prueba'.split(' '),
        'para ver si funciona lo que creo'.split(' '),
        'que va a funcionar ahora.'.split(' ')
    ])
    y = np.array([1,2,3])
    df = pd.concat((ds,pd.Series(y)),keys=['x','y'],axis=1)
    print(df)
    df = df.sort_values(by=['x'],key=lambda x: x.str.len(),ascending=False)
    print(df)

if __name__ == '__main__':
    #main()
    for i in (1,):
        print(i)