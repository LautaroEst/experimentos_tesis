import experiment as ex
import os
from datetime import datetime
import json
import torch
import torch.nn as nn
from accelerate import Accelerator
import numpy as np
from copy import deepcopy


def setup_logdir(rootdir,hyperparams,frac):
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    logdir = os.path.join(rootdir,"frac{:.3f}".format(frac))
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    
    with open(os.path.join(logdir,"hyperparams.json"),"w") as f:
        json.dump(hyperparams,f,indent=4,separators=(",",": "))

    return logdir


def pretrain(hyperparams,nclasses,logdir,melisa_frac):
    
    print("Loading data for pretrain...")
    train_dataloader, dev_dataloader, tokenizer = ex.load_data_for_pretrain(
        nclasses=nclasses,
        train_batch_size=hyperparams["train_batch_size"],
        dev_batch_size=128,
        max_tokens=hyperparams["max_tokens"],
        freq_cutoff=hyperparams["freq_cutoff"],
        max_sent_len=hyperparams["max_sent_len"],
        frac=melisa_frac
    )

    print("Initializing model for pretrain...")
    model = ex.init_lstm_model(
        hidden_size=hyperparams["hidden_size"],
        embedding_dim=hyperparams["embedding_dim"],
        hidden_layers=hyperparams["hidden_layers"],
        num_embeddings=len(tokenizer.vocab),
        pad_idx=tokenizer.vocab[tokenizer.pad_token],
        dropout=hyperparams["dropout"],
        num_outs=nclasses
    )
    
    print("Starting to pretrain...")
    ex.train_model(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        optimization=hyperparams["optimization"],
        learning_rate=hyperparams["learning_rate"],
        num_epochs=hyperparams["num_epochs"],
        train_eval_every=max([1,int(100*melisa_frac)]),
        dev_eval_every=max([1,int(100*melisa_frac)])*10,
        logdir=logdir
    )
    tokenizer.to_json(os.path.join(logdir,"tokenizer_config.json"))
    print()


def finetune(pretrain_logdir,finetune_logdir,test_dataset,nclasses,hyperparams):

    print("Loading data for finetune...")
    train_dataloader, dev_dataloader, test_dataloader, tokenizer = ex.load_data_for_finetune(
        nclasses=nclasses,
        dataset=test_dataset,
        train_batch_size=hyperparams["train_batch_size"],
        dev_batch_size=128,
        max_tokens=hyperparams["max_tokens"],
        freq_cutoff=hyperparams["freq_cutoff"],
        max_sent_len=hyperparams["max_sent_len"],
    )

    print("Loding model for finetune...")
    model = ex.load_lstm_model_from_checkpoint(
        hidden_size=hyperparams["hidden_size"],
        embedding_dim=hyperparams["embedding_dim"],
        hidden_layers=hyperparams["hidden_layers"],
        tokenizer=tokenizer,
        checkpoint=pretrain_logdir,
        dropout=0.0,
        num_outs=nclasses
    )

    print("Starting to finetune...")
    ex.train_model(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        optimization=hyperparams["optimization"],
        learning_rate=hyperparams["learning_rate"],
        num_epochs=hyperparams["num_epochs"],
        train_eval_every=50, # CAMBIAR POR 10
        dev_eval_every=500, # CAMBIAR POR 100
        logdir=finetune_logdir
    )
    tokenizer.to_json(os.path.join(finetune_logdir,"tokenizer_config.json"))

    print("Evaluating in test...")
    criterion = nn.CrossEntropyLoss(reduce="sum")
    accelerator = Accelerator(split_batches=True)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    test_loss, test_f1, test_cm = ex.eval_loss_f1(model,test_dataloader,criterion,accelerator,plot_cm=True)
    # test_cm = np.array2string(test_cm, precision=1, separator=',', suppress_small=True,formatter={"int": int})
    with open(os.path.join(finetune_logdir,"test_results.json"), "w") as f:
        json.dump({
            "loss":  test_loss,
            "f1-score": test_f1,
            "confusion matrix": test_cm.tolist()
        },f,indent=4,separators=(",",": "))

hyperparams = dict(
    hidden_size=100,
    embedding_dim=300,
    hidden_layers=2,
    dropout=0.1, 
    learning_rate=5e-4, 
    train_batch_size=16,
    num_epochs=20, # CAMBIAR POR 20
    optimization="adam",
    max_tokens=60000,
    freq_cutoff=4,
    max_sent_len=128
)


if __name__ == "__main__":
    nclasses = 5
    # for num_epochs, frac in zip([30, 20, 10, 10, 10],[0.001, 0.01, 0.1, 0.5, 1.]):
    for num_epochs, frac in zip([30, 10, 10],[0.001, 0.1, 1.]):
        # my_hyperparams = deepcopy(hyperparams)
        # my_hyperparams["num_epochs"] = num_epochs
        # pretrain_logdir = setup_logdir(rootdir="./pretrained_models",hyperparams=my_hyperparams,frac=frac)
        # pretrain(my_hyperparams,nclasses,logdir=pretrain_logdir,melisa_frac=frac)
        # for test_dataset in ["tass", "cine"]:
        pretrain_logdir =  "./pretrained_models/frac{:.3f}".format(frac)
        for test_dataset in ["amazon"]:
            finetune_logdir = setup_logdir(rootdir="./finetunned_{}_models".format(test_dataset),hyperparams=hyperparams,frac=frac)
            finetune(pretrain_logdir,finetune_logdir,test_dataset,nclasses,hyperparams)
