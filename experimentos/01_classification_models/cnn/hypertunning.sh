#! /bin/bash

declare -a dropoutarr=(0.0 0.2 0.5 0.7)

for dp in "${dropoutarr[@]}" 
do
    echo ""
    echo "Dropout: $dp"
    echo ""
    python main.py --nclasses 2 --lang es --devsize 0.05 --max_tokens 60000 --frequency_cutoff 1 --max_sent_len 512 --embedding_dim 300 --n_filters 8 --filter_sizes 3 5 7 --dropout $dp --num_epochs 8 --batch_size 256 --learning_rate 5e-4 --device "cuda:1" --eval_every 100
    echo ""
done
