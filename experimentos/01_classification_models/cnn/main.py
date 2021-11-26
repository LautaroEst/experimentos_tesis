from utils import load_data, train_dev_split, parse_arguments, show_results
from model import Classifier

RANDOM_SEED = 61273812

def load_and_split(lang,nclasses,devsize):
    df = load_data(
            lang=lang,
            split='train',
            nclasses=nclasses
        )#.iloc[:1000,:].reset_index(drop=True)
    df_train, df_dev = train_dev_split(df,devsize,RANDOM_SEED)
    return df_train, df_dev

def make_description(args,is_dev):
    description = """

Descripción del experimento:
----------------------------

Modelo de clasificación con una red neuronal convolucional.
La secuencia de vectores es convolucionada por grupos de n_filters 
filtros, cada uno de ellos con un tamaño distinto determinado por
el parámetro filter_sizes. Por ejemplo, si n_filters es 2 y 
filter_sizes es (2,3,4), la red convoluciona la entrada con 2 filtros
de tamaño 2, 2 de tamaño 3 y 2 de tamaño 4. A eso se le hace un pooling
por filtro y se concatena todo, resultando en un vector de tamaño
n_filters * len(filter_sizes), que pasa a un layer lineal.

Argumentos del entrenamiento utilizados:
- Cantidad de clases: {nclasses}
- Idioma: {lang}
- Cantidad máxima de palabras en el vocabulario: {max_tokens}
- Frecuencia mínima de cada palabra: {frequency_cutoff}
- Cantidad máxima de tokens por review: {max_sent_len}
- Dimensión de los embeddings: {embedding_dim}
- Tamaño de los filtros: {filter_sizes}
- Cantidad de canales por filtro: {n_filters}
- Probabilidad de dropout: {dropout}
- Tamaño del batch: {batch_size}
- Tasa de aprendizaje: {learning_rate}
- Cantidad de epochs: {num_epochs}
- Dispositivo de entrenamiento: {device}
- Mostrar los datos cada {eval_every} batches.
""".format(**args)

    if is_dev:
        description = "{}- Proporción utilizada para dev: {}\n"\
                        .format(description,args['devsize'])

    return description
    

def main():
    args = parse_arguments()
    is_dev = args['devsize'] is not None

    description = make_description(args,is_dev)
    lang = args.pop('lang')
    nclasses = args.pop('nclasses')
    _ = args.pop('test')
    devsize = args.pop('devsize')
    eval_every = args.pop('eval_every')
    
    if is_dev:
        df_train, df_dev_test = load_and_split(
                            lang=lang,
                            nclasses=nclasses,
                            devsize=devsize
                        )
    else:
        df_train = load_data(
            lang=lang,
            split='train',
            nclasses=nclasses
        )
        df_dev_test = load_data(
            lang=lang,
            split='test',
            nclasses=nclasses
        )#.iloc[:1000,:].reset_index(drop=True)

    # Model initialization:
    print('Initializing the model...')
    model = Classifier(nclasses,**args)

    # Model training:
    print('Training...')
    history = model.train(
                df_train['review_content'],
                df_train['review_rate'].values,
                eval_every=eval_every,
                dev=(df_dev_test['review_content'],df_dev_test['review_rate'].values)
            )

    # Model evaluation:
    print('Evaluating results...')
    y_train_predict = model.predict(df_train['review_content'])
    y_train_true = df_train['review_rate'].values
    y_devtest_predict = model.predict(df_dev_test['review_content'])
    y_devtest_true = df_dev_test['review_rate'].values
    show_results(
        y_train_predict,
        y_train_true,
        y_devtest_predict,
        y_devtest_true,
        history,
        nclasses,
        description,
        is_dev
    )
    
    

if __name__ == '__main__':
    main()