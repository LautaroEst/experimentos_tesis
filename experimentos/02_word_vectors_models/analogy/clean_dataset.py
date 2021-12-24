import pandas as pd
from tqdm import tqdm
import os


def main():
    data = {}
    with open("./questions-words_sp.txt","r") as f:
        for line in tqdm(f.readlines()):
            if line[0] == ":":
                key = line[2:-1]
                data[key] = {"w1": [], "w2": [], "w3": [], "w4": []}
                continue
            for i, w in enumerate(line[:-1].split(" "),1):
                data[key]["w"+str(i)].append(w)

    for dataset, values in data.items():
        pd.DataFrame(values).to_csv("./data/{}.csv".format(dataset),index=False)
            


    
    


if __name__ == "__main__":
    main()