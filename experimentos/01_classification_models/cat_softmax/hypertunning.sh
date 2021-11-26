#! /bin/bash

declare -a arr=(10 50 100 200)

for hs in "${arr[@]}" 
do
    echo "hidden size: $hs"
    echo ""
    python main.py --nclasses 5 --lang es --devsize 0.05 --ngram_range 1 2 --max_features 100000 --hidden_size $hs --num_epochs 16 --batch_size 512 --learning_rate 5e-4 --weight_decay 0.0 --device "cuda:1" --eval_every 100
    echo ""
done
