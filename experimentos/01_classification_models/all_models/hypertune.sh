#! /bin/bash

declare -a dropoutarray=(0.0 0.2 0.5 0.7)

for dp in "${dropoutarray[@]}" 
do
    echo ""
    echo "Dropout: $dp"
    echo ""
    python main.py --model "beto-uncased" --dataset "melisa-2" --devsize 0.05 --dropout $dp --num_epochs 1 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
    echo ""
done