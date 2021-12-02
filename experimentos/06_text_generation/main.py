import utils
from utils.data import load_melisa, load_amazon, load_and_split_melisa, normalize_dataset
from utils.io import parse_args

import pandas as pd


def load_dataset(dataset,devsize,split):
    print('Loading dataset...')
    if dataset == 'melisa':
        if split == 'dev':
            df_train, df_devtest = load_and_split_melisa(devsize)
        else:
            df_train = load_melisa('train')
            df_devtest = load_melisa('test')

    elif dataset == 'amazon':
        if split == 'dev':
            df_train = load_amazon(split='train')
            df_devtest = load_amazon(split='dev')
        else:
            df_train = pd.concat([
                load_amazon(split='train'),
                load_amazon(split='dev')
                ],ignore_index=True
            )
            df_devtest = load_amazon(split='test')
    else:
        raise NameError('Dataset not supported')

    print('Normalizing dataset...')
    ds_src_train = normalize_dataset(df_train['review_content'])
    ds_target_train = normalize_dataset(df_train['review_title'])
    ds_src_devtest = normalize_dataset(df_devtest['review_content'])
    ds_target_devtest = normalize_dataset(df_devtest['review_title'])
    data = (ds_src_train, ds_target_train, ds_src_devtest, ds_target_devtest)
    return data


def main():
    dataset_args, model_name, model_args, eval_every = parse_args()
    data = load_dataset(**dataset_args)
    print(data[0])


if __name__ == '__main__':
    main()