#! /bin/bash

declare -a modelsarray=("beto-uncased" "mbert-sent" "xlm-roberta-sent" "electricidad")

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
    echo "tass-5. Model: $m"
    echo ""
    python main.py --model $m --dataset "tass-5" --test --dropout 0.0 --num_epochs 1 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
    echo ""
done


echo ""
echo "tass-2. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "tass-2" --test --dropout 0.0 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
echo ""
echo ""
echo "tass-3. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "tass-3" --test --dropout 0.0 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
echo ""
echo "tass-5. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "tass-5" --test --dropout 0.0 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
echo ""


