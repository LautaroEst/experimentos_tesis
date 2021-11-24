from utils import load_data, train_dev_split, parse_arguments, show_results
from model import Classifier

RANDOM_SEED = 61273812

def load_and_split(lang,nclasses,devsize):
    df = load_data(
            lang=lang,
            split='train',
            nclasses=nclasses
        )
    df_train, df_dev = train_dev_split(df,devsize,RANDOM_SEED)
    return df_train, df_dev

def make_description(args,is_dev):
    description = """

Descripción del experimento:
----------------------------

Modelo de clasificación con un modelo TfIdf + Naive Bayes.

Argumentos del entrenamiento utilizados:
- Cantidad de clases: {nclasses}
- Idioma: {lang}
- Rango de los n-gramas: {ngram_range}
- Cantidad máxima de features en el vocabulario: {max_features}
""".format(**args)

    if is_dev:
        description = "{}- Proporción utilizada para dev: {}\n"\
                        .format(description,args['devsize'])

    return description
    

def main():
    args = parse_arguments()
    is_dev = args['devsize'] is not None

    description = make_description(args,is_dev)
    print(description)
    lang = args.pop('lang')
    nclasses = args.pop('nclasses')
    _ = args.pop('test')
    devsize = args.pop('devsize')
    
    if is_dev:
        df_train, df_dev_test = load_and_split(
                            lang=lang,
                            nclasses=nclasses,
                            devsize=devsize
                        )
    else:
        df_dev_test = load_data(
            lang=lang,
            split='test',
            nclasses=nclasses
        )

    # Model initialization:
    print('Initializing the model...')
    model = Classifier(nclasses,**args)

    # Model training:
    print('Training...')
    model.train(
            df_train['review_content'],
            df_train['review_rate'].values
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
        nclasses,
        description,
        is_dev
    )
    
    

if __name__ == '__main__':
    main()