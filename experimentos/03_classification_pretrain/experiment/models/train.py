import os
import torch
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from sklearn.metrics import f1_score
import numpy as np

def dev_eval(model,dev_dataloader,criterion,accelerator):
    if model.training:
        was_training = True
    
    cum_loss = cum_num_examples = 0
    cum_predictions = []
    cum_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dev_dataloader:
            labels = batch.pop("label")
            scores = model(**batch)
            
            loss = criterion(scores,labels)

            cum_loss += loss.item()
            cum_num_examples += len(labels)
            _, predictions = torch.max(accelerator.gather(scores),dim=-1)
            cum_predictions.append(predictions.detach().cpu().view(-1).numpy())
            cum_labels.append(accelerator.gather(labels).detach().cpu().view(-1).numpy())

    dev_loss = cum_loss/cum_num_examples
    dev_f1 = f1_score(np.hstack(cum_labels),np.hstack(cum_predictions),average="macro")
    print("Dev loss: {:.4f}. Dev f1-macro: {:.2f}".format(dev_loss,dev_f1),end="\n\n")

    if was_training:
        model.train()

    return dev_loss, dev_f1

def train_model(
        model,
        train_dataloader,
        dev_dataloader,
        optimization,
        learning_rate,
        num_epochs,
        train_eval_every,
        dev_eval_every,
        logdir
    ):

    writer = SummaryWriter(log_dir=logdir)
    # example_batch = next(iter(train_dataloader))
    # example_labels = example_batch.pop("label")
    # writer.add_graph(model,[example_batch["input_ids"],example_batch["attention_mask"]])
    
    if optimization == "adam":
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    else:
        raise TypeError("Optimizer procedure not supported")

    accelerator = Accelerator(split_batches=True)
    model, train_dataloader, dev_dataloader, optimizer = accelerator.prepare(
                        model, train_dataloader, dev_dataloader, optimizer
    )

    criterion = nn.CrossEntropyLoss(reduce="sum")
    num_batches = len(train_dataloader)
    cum_loss = cum_num_examples = 0
    cum_predictions = []
    cum_labels = []
    dev_f1_history = []
    for e in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            labels = batch.pop("label")
            scores = model(**batch)
            
            loss = criterion(scores,labels)
            accelerator.backward(loss)

            cum_loss += loss.item()
            cum_num_examples += len(labels)
            _, predictions = torch.max(accelerator.gather(scores),dim=-1)
            cum_predictions.append(predictions.detach().cpu().view(-1).numpy())
            cum_labels.append(accelerator.gather(labels).detach().cpu().view(-1).numpy())

            optimizer.step()

            if (e * num_batches + i) % train_eval_every == 0:
                avg_loss = cum_loss/cum_num_examples
                avg_f1 = f1_score(np.hstack(cum_labels),np.hstack(cum_predictions),average="macro")
                print("Epoch {}/{}. Batch {}/{}. Avg. train loss: {:.4f}. Avg f1-macro: {:.2f}".format(
                    e,num_epochs,i,num_batches,avg_loss,avg_f1))
                writer.add_scalar("train avg loss", avg_loss, e * num_batches + i)
                writer.add_scalar("train avg f1-score", avg_f1, e * num_batches + i)
                cum_loss = cum_num_examples = 0
                cum_predictions = []
                cum_labels = []

            if (e * num_batches + i) % dev_eval_every == 0:
                print("Evaluating on dev...")
                dev_loss, dev_f1 = dev_eval(model,dev_dataloader,criterion,accelerator)
                writer.add_scalar("dev loss", dev_loss, e * num_batches + i)
                writer.add_scalar("dev f1-score", dev_f1, e * num_batches + i)
                if len(dev_f1_history) == 0 or dev_f1 > max(dev_f1_history):
                    save_checkpoint(logdir,model,optimizer)

    writer.close()

def save_checkpoint(logdir,model,optimizer):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },os.path.join(logdir,"best_model_checkpoint.pkl"))
