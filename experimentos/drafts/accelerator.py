from accelerate import Accelerator
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import pandas as pd


class AmazonDataset(Dataset):

    def __init__(self):
        dataset = load_dataset("amazon_reviews_multi","es")
        df = pd.DataFrame(dataset["train"]).loc[:,['review_body','review_title','stars']].sample(frac=1,random_state=RANDOM_SEED).reset_index(drop=True)
        self.df = df.rename(columns={'review_body':'content', 'stars': 'label'})
        self.tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
        
    def __getitem__(self,idx):
        sample = self.df.iloc[idx,:]
        content, label = sample["content"].values, sample["label"].values
        return content, label

    def __len__(self):
        return len(self.df)    



def main():
    # train_dataloader = DataLoader(
    #     TensorDataset(
    #         torch.randn(1000,10,dtype=torch.float32),
    #         torch.randint(0,5,(1000,),dtype=torch.long)
    #     ),
    #     batch_size=32,
    #     shuffle=True
    # )

    train_dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
    )
    

    model = BertForSequenceClassification.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
    optimizer = optim.Adam(model.parameters(),lr=1e-5)

    accelerator = Accelerator(split_batches=False)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    # criterion = nn.CrossEntropyLoss()

    epochs = 4
    for e in range(epochs):
        for i, batch in enumerate(train_dataloader):

            x, y_true = batch
            print(x.size())
            print(x.device)
            # y_pred = model(x)
            # loss = criterion(y_pred,y_true)
            accelerator.backward(loss)
            print("train loss: {:.4f}".format(loss.item()))
            name, param = next(model.named_parameters())
            print("graident norm ({}): {:.4f}".format(name,torch.norm(param.grad.data)))
            print()








if __name__ == "__main__":
    main()