import pickle
import torch

def main():
    # with open("results/2021-12-13-20-38-38/ppl_history.pkl","rb") as f:
    #     history = pickle.load(f)

    # print(history["dev_loss_history"][-20:])
    # print(history["dev_f1score_history"][-20:])
    checkpoint = torch.load("results/amazon_none_cbow/checkpoint.pkl")
    print(checkpoint['optimizer_state_dict'])

if __name__ == "__main__":
    main()