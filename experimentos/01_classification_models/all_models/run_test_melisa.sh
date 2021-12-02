#! /bin/bash

# declare -a modelsarray=("beto-uncased" "electricidad" "mbert-sent")

# for m in "${modelsarray[@]}"
# do
#     echo ""
#     echo "Melisa-2. Model: $m"
#     echo ""
#     python main.py --model $m --dataset "melisa-2" --test --dropout 0.05 --num_epochs 1 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
#     echo ""
#     echo ""
#     echo "Melisa-3. Model: $m"
#     echo ""
#     python main.py --model $m --dataset "melisa-3" --test --dropout 0.05 --num_epochs 1 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
#     echo ""
#     echo "Melisa-5. Model: $m"
#     echo ""
#     python main.py --model $m --dataset "melisa-5" --test --dropout 0.05 --num_epochs 1 --batch_size 8 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
# done

echo ""
echo "Melisa-2. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "melisa-2" --test --dropout 0.05 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
echo ""
echo ""
echo "Melisa-3. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "melisa-3" --test --dropout 0.05 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100
echo ""
echo "Melisa-5. Model: gtp2-esp"
echo ""
python main.py --model "gtp2-esp" --dataset "melisa-5" --test --dropout 0.05 --num_epochs 1 --batch_size 2 --learning_rate 1e-5 --device "cuda:1" --eval_every 100

