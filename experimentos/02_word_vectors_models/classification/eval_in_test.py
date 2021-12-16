from main import *


def load_dataset(nclasses,dataset):
    if dataset == 'melisa':
        df_train = load_melisa(split='train',nclasses=5)
        df_test = load_melisa(split='test',nclasses=5)
        ds_train = normalize_dataset(df_train['review_content'])
        ds_test = normalize_dataset(df_test['review_content'])
        y_train = df_train['review_rate']
        y_test = df_test['review_rate']
        
    elif dataset == 'amazon':
        df_train = pd.concat(
                [load_amazon(split='train',nclasses=nclasses),
                load_amazon(split='dev',nclasses=nclasses)],
                ignore_index=True
        )
        df_test = load_amazon(split='test',nclasses=nclasses)
        ds_train = normalize_dataset(df_train['review_content'])
        ds_test = normalize_dataset(df_test['review_content'])
        y_train = df_train['review_rate']
        y_test = df_test['review_rate']

    elif dataset == 'tass':
        df_train = load_tass(split='train',nclasses=5)
        df_test = load_tass(split='test',nclasses=5)
        ds_train = normalize_dataset(df_train['tweet'])
        ds_test = normalize_dataset(df_test['tweet'])
        y_train = df_train['label']
        y_test = df_test['label']

    elif dataset == 'cine':
        df_train = pd.concat(
                [load_cine(split='train',nclasses=nclasses),
                load_cine(split='dev',nclasses=nclasses)],
                ignore_index=True
        )
        df_test = load_cine(split='test',nclasses=nclasses)

        def limit_len(ds):
            ds = ds.apply(lambda s: s[:2000])
            return ds

        ds_train = limit_len(normalize_dataset(df_train['review_content']))
        ds_test = limit_len(normalize_dataset(df_test['review_content']))
        y_train = df_train['review_rate']
        y_test = df_test['review_rate']
    
    else:
        raise NameError("Dataset not supported")
            
    data = {
        'sent_train': ds_train, 
        'y_train': y_train, 
        'sent_test': ds_test, 
        'y_test': y_test
    }
    return data

def main():
    
    # Read args
    print("Parsing args...")
    args = parse_args()
        
    # Dataset loading:
    print("Loading dataset...")
    dataset_args = args.pop('dataset_args')
    dataset_args.pop('devsize')
    data = load_dataset(nclasses=5,**dataset_args)
    # data = {key: val[:1000] for key, val in data.items()}

    # Inicialización del tokenizer
    print("Initializing tokenizer...")
    tokenizer_kwargs = args.pop('tokenizer_kwargs')
    tokenizer = init_tokenizer(data['sent_train'],tokenizer_kwargs)
    
    # Inicialización del modelo
    print("Initializing model...")
    model_kwargs = args.pop('model_kwargs')
    model = init_model(model_kwargs,tokenizer)
    print("Training...")
    history = train(
        model=model,
        data=data,
        results_dir=args['results_dir'],
        **args['train_kwargs']
    )

    # Gráfico del historial
    print("Plotting history...")
    plot_history(history,args['results_dir'])

    # Evaluación de los resultados
    print("Evaluating results...")
    evaluate_all(model,data,args['results_dir'],history,**args['train_kwargs'])



if __name__ == "__main__":
    main()