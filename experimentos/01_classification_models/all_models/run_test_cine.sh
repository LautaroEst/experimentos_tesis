#! /bin/bash

declare -a modelsarray=("beto-uncased" "mbert-sent" "electricidad")

for m in "${modelsarray[@]}"
do
    echo ""
    echo "cine-2. Model: $m"
    echo ""
    python main.py --model $m --dataset "cine-2" --test --dropout 0.0 --num_epochs 2 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 10
    echo ""
    echo ""
    echo "cine-3. Model: $m"
    echo ""
    python main.py --model $m --dataset "cine-3" --test --dropout 0.0 --num_epochs 2 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 10
    echo ""
    echo "cine-5. Model: $m"
    echo ""
    python main.py --model $m --dataset "cine-5" --test --dropout 0.0 --num_epochs 2 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 10
    echo ""
done


echo ""
echo "cine-2. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "cine-2" --test --dropout 0.0 --num_epochs 2 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 10
echo ""
echo ""
echo "cine-3. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "cine-3" --test --dropout 0.0 --num_epochs 2 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 10
echo ""
echo "cine-5. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "cine-5" --test --dropout 0.0 --num_epochs 2 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 10
echo ""


