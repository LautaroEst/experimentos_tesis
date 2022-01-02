import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
import torch
import json
import re


def init_result_dict():
    results_dict = {}
    for dataset in ["amazon", "cine", "melisa", "tass"]:
        results_dict[dataset] = {}
        for wordvector in ["none", "word2vec", "glove", "fasttext", "elmo"]:
            results_dict[dataset][wordvector] = {}
            for model in ["cbow", "lstm", "cnn"]:
                results_dict[dataset][wordvector][model] = {"f1": None, "result_dir": None}
    return results_dict


def make_table(results_dirs):

    results_dict = init_result_dict()

    for result_dir in results_dirs:

        with open(os.path.join(result_dir,"config.json"),"r") as f:
            config = json.load(f)

        dataset = config['dataset']
        wordvector = "none" if config['model']['embeddings'] is None else re.findall(r"(none|word2vec|fasttext|glove|elmo)",config['model']['embeddings'])[0]
        model = config['model']['name']
        
        try:
            with open(os.path.join(result_dir,"results.txt"),"r") as f:
                count = 1
                for line in f.readlines():
                    if "macro avg" in line:
                        count += 1
                        if count == 2:
                            break

                current_value = results_dict[dataset][wordvector][model]["f1"]
                f1_score = float(re.split(r"\s+",line)[5])*100
                if current_value is None or current_value < f1_score:
                    results_dict[dataset][wordvector][model]["f1"] = f1_score
                    results_dict[dataset][wordvector][model]["result_dir"] = result_dir
        except FileNotFoundError:
            pass

    first_dataset = next(iter(results_dict.keys()))
    rows = [" & " + " & ".join([dataset for dataset in results_dict.keys()]) + "\\\\\n"]
    for wordvector in results_dict[first_dataset].keys():
        for model in results_dict[first_dataset][wordvector].keys():
            row = "{}+{}".format(model,wordvector)
            for dataset in results_dict.keys():
                f1_score = results_dict[dataset][wordvector][model]["f1"]
                rd = results_dict[dataset][wordvector][model]["result_dir"]
                if f1_score is not None:
                    row = "{} & {:.2f}({}) ".format(row,f1_score,rd)
                else:
                    row = "{} & - ".format(row)
            row = "{} \\\\\n".format(row)
            rows.append(row)
        row = "{}\\hline\n".format(row[:-1])
        rows[-1] = row

    return rows
    
    

def main():
    
    results_root = "./results"
    results_dirs = []
    models_dirs = os.listdir(results_root)
    
    for model_dir in models_dirs:
        path = os.path.join(results_root,model_dir)
        if os.path.isdir(path):
            results_dirs.extend([os.path.join(path, result_path) for result_path in os.listdir(path)])
    
    rows = make_table(results_dirs)

    with open(os.path.join(results_root,"results_table.txt"),"w") as f:
        for row in rows:
            f.write(row)

        
    


                


if __name__ == "__main__":
    main()