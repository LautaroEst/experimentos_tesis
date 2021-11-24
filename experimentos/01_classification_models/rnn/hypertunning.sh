#! /bin/bash

declare -a arr=(10000 20000 40000 60000 80000)

for mf in "${arr[@]}" 
do
    echo "max features: $mf"
    echo ""
    python main.py --nclasses 2 --lang es --devsize 0.05 --ngram_range 1 2 --max_features 100 --hidden_size 1 --num_epochs 1 --batch_size 1024 --learning_rate 1e-3 --weight_decay 0.0 --device "cuda:1" --eval_every 100
    echo ""
done
