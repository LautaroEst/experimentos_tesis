import pickle
import torch
from utils.data import load_tass, load_cine, load_and_split_tass, load_and_split_melisa, load_melisa

def main():
    # with open("results/2021-12-13-20-38-38/ppl_history.pkl","rb") as f:
    #     history = pickle.load(f)

    # print(history["dev_loss_history"][-20:])
    # print(history["dev_f1score_history"][-20:])
    # checkpoint = torch.load("results/amazon_none_cbow/checkpoint.pkl")
    # print(checkpoint['optimizer_state_dict'])
    print("train")
    train, dev = load_cine(split="train",nclasses=2), load_cine(split="dev",nclasses=2)
    print(len(train),len(dev))
    print()
    print("test")
    dataset = load_cine(split='test',nclasses=2)
    print(len(dataset))

if __name__ == "__main__":
    main()