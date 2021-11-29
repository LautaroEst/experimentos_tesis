#! /bin/bash

declare -a modelsarray=("beto-uncased" "mbert-sent" "xlm-roberta-sent" "gtp2-esp" "electricidad")

for m in "${modelsarray[@]}"
do
    echo ""
    echo "tass-2. Model: $m"
    echo ""
    python main.py --model $m --dataset "tass-2" --test --dropout 0.0 --num_epochs 1 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
    echo ""
    echo ""
    echo "tass-3. Model: $m"
    echo ""
    python main.py --model $m --dataset "tass-3" --test --dropout 0.0 --num_epochs 1 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
    echo ""
done



