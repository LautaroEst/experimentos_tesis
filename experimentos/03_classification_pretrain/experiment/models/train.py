from io import BytesIO
import json
import os
import torch
from torch import optim, nn
from torchvision.transforms import ToTensor
from PIL import Image
from accelerate import Accelerator
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import product

def eval_loss_f1(model,dataloader,criterion,accelerator,plot_cm=False):
    if model.training:
        was_training = True
    
    cum_loss = cum_num_examples = 0
    cum_predictions = []
    cum_labels = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("label")
            scores = model(**batch)
            
            loss = criterion(scores,labels)

            cum_loss += loss.item()
            cum_num_examples += len(labels)
            _, predictions = torch.max(accelerator.gather(scores),dim=-1)
            cum_predictions.append(predictions.detach().cpu().view(-1).numpy())
            cum_labels.append(accelerator.gather(labels).detach().cpu().view(-1).numpy())

    loss = cum_loss/cum_num_examples
    cum_labels = np.hstack(cum_labels)
    cum_predictions = np.hstack(cum_predictions)
    f1 = f1_score(cum_labels,cum_predictions,average="macro")
    cm = confusion_matrix(cum_labels,cum_predictions)

    if was_training:
        model.train()

    if plot_cm:
        return loss, f1, cm
    else:
        return loss, f1

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

    if optimization == "adam":
        optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    elif optimization == "sgd":
        optimizer = optim.SGD(model.parameters(),lr=learning_rate)
    elif optimization == "rms":
        optimizer = optim.RMSprop(model.parameters(),lr=learning_rate)
    else:
        raise TypeError("Optimizer procedure not supported")

    accelerator = Accelerator(split_batches=True)
    model, train_dataloader, dev_dataloader, optimizer = accelerator.prepare(
                        model, train_dataloader, dev_dataloader, optimizer
    )

    criterion = nn.CrossEntropyLoss(reduce="sum")
    num_batches = len(train_dataloader)
    cum_loss = cum_num_examples = cum_cum_num_examples = 0
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
            cum_cum_num_examples += cum_num_examples
            _, predictions = torch.max(accelerator.gather(scores),dim=-1)
            cum_predictions.append(predictions.detach().cpu().view(-1).numpy())
            cum_labels.append(accelerator.gather(labels).detach().cpu().view(-1).numpy())

            optimizer.step()

            if ((e * num_batches + i) % train_eval_every == 0) or (e == num_epochs-1 and i == num_batches-1):
                avg_loss = cum_loss/cum_num_examples
                avg_f1 = f1_score(np.hstack(cum_labels),np.hstack(cum_predictions),average="macro")
                print("Epoch {}/{}. Batch {}/{}. Avg. train loss: {:.4f}. Avg f1-macro: {:.2f}".format(
                    e+1,num_epochs,i+1,num_batches,avg_loss,avg_f1))
                writer.add_scalar("Loss/train", avg_loss, cum_cum_num_examples)
                writer.add_scalar("F1-score/train", avg_f1, cum_cum_num_examples)
                cum_loss = cum_num_examples = 0
                cum_predictions = []
                cum_labels = []

            if ((e * num_batches + i) % dev_eval_every == 0) or (e == num_epochs-1 and i == num_batches-1):
                print("Evaluating on dev...")
                dev_loss, dev_f1 = eval_loss_f1(model,dev_dataloader,criterion,accelerator,plot_cm=False)
                print("Dev loss: {:.4f}. Dev f1-macro: {:.2f}".format(dev_loss,dev_f1),end="\n\n")
                writer.add_scalar("Loss/dev", dev_loss, cum_cum_num_examples)
                writer.add_scalar("F1-score/dev", dev_f1, cum_cum_num_examples)
                if len(dev_f1_history) == 0 or dev_f1 > max(dev_f1_history):
                    dev_f1_history.append(dev_f1)
                    save_checkpoint(writer.log_dir,model,optimizer)

    model = load_best_model(model,logdir,accelerator)
    print("Evaluating best model on train...")
    train_loss, train_f1, train_cm = eval_loss_f1(model,train_dataloader,criterion,accelerator,plot_cm=True)
    print("Evaluating best model on dev...")
    dev_loss, dev_f1, dev_cm = eval_loss_f1(model,dev_dataloader,criterion,accelerator,plot_cm=True)

    with open(os.path.join(logdir,"hyperparams.json"),"r") as f:
        hyperparams = json.load(f)

    writer.add_hparams(hyperparams,{
        "Loss/train": train_loss,
        "Loss/dev": dev_loss,
        "F1-score/train": train_f1,
        "F1-score/dev": dev_f1
    },run_name="hparams")

    train_cm = plot_confusion_matrix(train_cm)
    dev_cm = plot_confusion_matrix(dev_cm)
    writer.add_image("Confusion Matrix/train",train_cm)
    writer.add_image("Confusion Matrix/dev",dev_cm)

    writer.close()


def save_checkpoint(logdir,model,optimizer):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    },os.path.join(logdir,"best_model_checkpoint.pkl"))


def load_best_model(model,log_dir,accelerator):
    model.cpu()
    state_dict = torch.load(os.path.join(log_dir,"best_model_checkpoint.pkl"))["model_state_dict"]
    model.load_state_dict(state_dict)
    model = accelerator.prepare(model)
    return model


def plot_confusion_matrix(cm):
    nclasses = cm.shape[0]
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ticks_marks = list(range(nclasses))
    ax.set_xticks(ticks_marks)
    ax.set_xticklabels(["{}".format(int(t+1)) for t in ticks_marks],fontsize='xx-large')
    ax.set_yticks(ticks_marks)
    ax.set_yticklabels(["{}".format(int(t+1)) for t in ticks_marks],fontsize='xx-large')
    
    cm = np.around(cm / cm.sum(), decimals=2)
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    threshold = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        ax.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    
    ax.set_title("Confusion Matrix",fontsize='xx-large')
    ax.set_ylabel('True label',fontsize='xx-large')
    ax.set_xlabel('Predicted label',fontsize='xx-large')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation="vertical")
    fig.tight_layout()
    
    buf = BytesIO()
    transform = ToTensor()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    img = transform(Image.open(buf))
    return img


