#! /bin/bash

declare -a modelsarray=("gtp2-esp")

for m in "${modelsarray[@]}"
do
    echo ""
    echo "Melisa-2. Model: $m"
    echo ""
    python main.py --model $m --dataset "melisa-2" --test --dropout 0.0 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
    echo ""
    echo ""
    echo "Melisa-3. Model: $m"
    echo ""
    python main.py --model $m --dataset "melisa-3" --test --dropout 0.0 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
    echo ""
    echo "Melisa-5. Model: $m"
    echo ""
    python main.py --model $m --dataset "melisa-5" --test --dropout 0.0 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
done



