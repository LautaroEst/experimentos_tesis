import json
import os
import torch
import torch.nn as nn
from accelerate import Accelerator
import experiment as ex



def zero_shot(pretrained_logdir,hyperparams,dataset,nclasses):
    criterion = nn.CrossEntropyLoss(reduce="sum")
    accelerator = Accelerator(split_batches=True)

    print("Loading data for zero shot...")
    _, _, test_dataloader, tokenizer = ex.load_data_for_finetune(
        nclasses=nclasses,
        dataset=dataset,
        train_batch_size=hyperparams["train_batch_size"],
        dev_batch_size=128,
        max_tokens=hyperparams["max_tokens"],
        freq_cutoff=hyperparams["freq_cutoff"],
        max_sent_len=hyperparams["max_sent_len"],
    )

    print("Loding model for zero shot...")
    model = ex.load_lstm_model_from_checkpoint(
        hidden_size=hyperparams["hidden_size"],
        embedding_dim=hyperparams["embedding_dim"],
        hidden_layers=hyperparams["hidden_layers"],
        tokenizer=tokenizer,
        checkpoint=pretrained_logdir,
        dropout=0.0,
        num_outs=nclasses
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    test_loss, test_f1, test_cm = ex.eval_loss_f1(model,test_dataloader,criterion,accelerator,plot_cm=True)
    # test_cm = np.array2string(test_cm, precision=1, separator=',', suppress_small=True,formatter={"int": int})
    return {
        "loss":  test_loss,
        "f1-score": test_f1,
        "confusion matrix": test_cm.tolist()
    }

def main(hyperparams):
    nclasses = 5
    for frac in [1.]:#[0.001, 0.01, 0.1, 0.5, 1.]:
        pretrained_logdir = "pretrained_models/frac{:.3f}".format(frac)
        zero_shot_results = {}
        for test_dataset in ["tass", "cine", "amazon"]:
            zero_shot_results[test_dataset] = zero_shot(pretrained_logdir,hyperparams,test_dataset,nclasses)
        with open("zero_shot_results/zero_shot_results_frac{:.3f}.json".format(frac), "w") as f:
            json.dump(zero_shot_results,f,indent=4,separators=(",",": "))


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
    main(hyperparams)