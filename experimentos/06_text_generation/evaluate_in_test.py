from utils.io import read_results, load_model
from utils.data import load_amazon, load_melisa, normalize_dataset
import argparse
import pandas as pd
from models import greedy_decoding
import os

def load_test_data(dataset):
    if dataset == 'melisa':
        df = load_melisa(split="test")
    elif dataset == 'amazon':
        df = load_amazon(split="test")
    ds_src = normalize_dataset(df['review_content'])
    ds_target = normalize_dataset(df['review_title'])
    data = {
        'src': ds_src, 
        'tgt': ds_target, 
    }
    return data

def save_sents_to_file(tgt_sents,pred_sents,path):

    df = pd.concat([
        pd.Series(tgt_sents),
        pd.Series(pred_sents)
    ], ignore_index=True, axis=1)
    df = df.rename(columns={0: "True", 1: "Predicted"})
    df.to_csv(path,index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',type=str,required=True)
    results_dir = vars(parser.parse_args())['results_dir']
    results = read_results(results_dir)
    model = load_model(results)
    data = load_test_data(results['config']['dataset'])

    tgt_tokenizer = results['tgt_tokenizer']
    pred_sents = greedy_decoding(model,data['src'],tgt_tokenizer,results['config']['eval_kwargs']['max_len_of_pred_sent'])
    sents_path_file = os.path.join(results_dir,"test_results.csv")
    target_sents = [tgt_tokenizer.tokenize(sent) for sent in data['tgt']]
    save_sents_to_file(target_sents,pred_sents,sents_path_file)

    

if __name__ == "__main__":
    main()