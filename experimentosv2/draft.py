import pandas as pd

def main():
    ds = pd.Series([
        'Esto es una prueba'.split(' '),
        'para ver si funciona lo que creo'.split(' '),
        'que va a funcionar ahora.'.split(' ')
    ])
    ds = ds.sort_values(key=lambda x: x.str.len(),ascending=False)
    print(ds)

if __name__ == '__main__':
    main()