from torch.utils.tensorboard import SummaryWriter
import json
import os
import experiment as ex
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np


def load_model(log_dir):
    with open(os.path.join(log_dir,"hyperparams.json"),"r") as f:
        hyperparams = json.load(f)
    tokenizer = ex.WordTokenizer.from_json(os.path.join(log_dir,"tokenizer_config.json"))
    model = ex.init_lstm_model(
        hyperparams["hidden_size"],
        hyperparams["embedding_dim"],
        len(tokenizer.vocab),
        tokenizer.vocab[tokenizer.pad_token],
        hyperparams["dropout"],
        5
    )
    # with open(os.path.join(log_dir,"best_model_checkpoint.pkl"), "r") as f:
    #     state_dict = torch.load(f)["model_state_dict"]
    state_dict = torch.load(os.path.join(log_dir,"best_model_checkpoint.pkl"))["model_state_dict"]
    model.load_state_dict(state_dict)
    return model


def evaluate_split(model,dataloader,device):

    cum_loss = cum_num_examples = 0
    cum_predictions = []
    cum_labels = []

    criterion = torch.nn.CrossEntropyLoss(reduce="sum")

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            labels = batch.pop("label").to(device=device)
            scores = model(**{key: val.to(device=device) for key, val in batch.items()})

            loss = criterion(scores,labels)

            cum_loss += loss.item()
            cum_num_examples += len(labels)
            _, predictions = torch.max(scores,dim=-1)
            cum_predictions.append(predictions.cpu().view(-1).numpy())
            cum_labels.append(labels.cpu().view(-1).numpy())

        loss = cum_loss/cum_num_examples
        f1 = f1_score(np.hstack(cum_labels),np.hstack(cum_predictions),average="macro")
    
    return loss, f1

def main(log_dir):
    writer = SummaryWriter(log_dir=log_dir)
    with open(os.path.join(log_dir,"hyperparams.json"),"r") as f:
        hyperparams = json.load(f)
    model =  load_model(log_dir)
    device = torch.device("cuda:1")
    model.to(device)
    train_dataloader, dev_dataloader, _ = ex.load_data_for_pretrain(
        nclasses=5,
        train_batch_size=hyperparams["train_batch_size"],
        dev_batch_size=64,
        max_tokens=hyperparams["max_tokens"],
        freq_cutoff=hyperparams["freq_cutoff"],
        max_sent_len=hyperparams["max_sent_len"]
    )
    train_loss, train_f1 = evaluate_split(model,train_dataloader,device)
    dev_loss, dev_f1 = evaluate_split(model,dev_dataloader,device)

    writer.add_hparams(hyperparams,{
        "train avg loss": train_loss,
        "dev loss": dev_loss,
        "train avg f1-score": train_f1,
        "dev f1-score": dev_f1
    })
    writer.close()



if __name__ == "__main__":
    # for log_dir in os.listdir("./results_pretrain/"):
    #     main(os.path.join("./results_pretrain/",log_dir))
    # writer = SummaryWriter(log_dir="./results_pretrain")
    # print(dir(writer))
    # print(writer.log_dir)
    # writer.close()
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image

    buf = BytesIO()
    transform = ToTensor()
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    ax.plot([1,2,3,4,5])

    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = transform(Image.open(buf))
    print(img)


    


    
