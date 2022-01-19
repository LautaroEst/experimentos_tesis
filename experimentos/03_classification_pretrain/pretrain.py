import numpy as np
import experiment as ex
import os
from datetime import datetime
import json
from itertools import product


hyperparams = dict(
    hidden_size=[200],
    embedding_dim=[300],
    dropout=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
    learning_rate=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5], 
    train_batch_size=[8, 16, 32, 64, 128, 256, 512],
    num_epochs=[1, 2, 5, 10, 20, 50],
    optimization=["adam"],
    max_tokens=[40000, 60000, 80000],
    freq_cutoff=[1],
    max_sent_len=[128]
)

def setup_logdir(rootdir,hyperparams):
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    logdir = os.path.join(rootdir,datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(logdir)
    
    with open(os.path.join(logdir,"hyperparams.json"),"w") as f:
        json.dump(hyperparams,f,indent=4,separators=(",",": "))

    return logdir


def main(hyperparams):
    nclasses = 5
    logdir = setup_logdir(rootdir="./results_pretrain",hyperparams=hyperparams)
    print("Loading data...")
    train_dataloader, dev_dataloader, tokenizer = ex.load_data_for_pretrain(
        nclasses=nclasses,
        train_batch_size=hyperparams["train_batch_size"],
        dev_batch_size=64,
        max_tokens=hyperparams["max_tokens"],
        freq_cutoff=hyperparams["freq_cutoff"],
        max_sent_len=hyperparams["max_sent_len"]
    )
    print("Initializing model...")
    model = ex.init_lstm_model(
        hidden_size=hyperparams["hidden_size"],
        embedding_dim=hyperparams["embedding_dim"],
        num_embeddings=len(tokenizer.vocab),
        pad_idx=tokenizer.vocab[tokenizer.pad_token],
        dropout=hyperparams["dropout"],
        num_outs=nclasses
    )
    print("Starting to train...")
    ex.train_model(
        model=model,
        train_dataloader=train_dataloader,
        dev_dataloader=dev_dataloader,
        optimization=hyperparams["optimization"],
        learning_rate=hyperparams["learning_rate"],
        num_epochs=hyperparams["num_epochs"],
        train_eval_every=100,
        dev_eval_every=1000,
        logdir=logdir
    )

    tokenizer.to_json(os.path.join(logdir,"tokenizer_config.json"))


if __name__ == "__main__":
    num_trains = 10
    rs = np.random.RandomState(173485)
    keys, vals = zip(*list(hyperparams.items()))
    samples = list(product(*vals))
    rndm_idx = rs.permutation(len(samples))[:num_trains]
    for idx in rndm_idx:
        sample_hyperparams = {key: val for key, val  in zip(keys,samples[idx])}
        main(sample_hyperparams)