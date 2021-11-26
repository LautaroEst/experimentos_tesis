#! /bin/bash

declare -a hiddenarray=(100 200)
declare -a numlayerarray=(1 2 4)

for hs in "${hiddenarray[@]}"
do
    for nl in "${numlayerarray[@]}" 
    do
        echo ""
        echo "Hidden size: $hs. Num layers: $nl"
        echo ""
        python main.py --nclasses 5 --lang es --devsize 0.05 --max_tokens 60000 --frequency_cutoff 1 --max_sent_len 512 --embedding_dim 300 --hidden_size $hs --num_layers $nl --dropout 0.0 --num_epochs 16 --batch_size 256 --learning_rate 5e-4 --device "cuda:1" --eval_every 100
        echo ""
    done
done
