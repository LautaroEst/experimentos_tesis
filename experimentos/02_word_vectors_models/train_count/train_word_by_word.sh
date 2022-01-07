#! /bin/bash

declare -a reweightarray=("none" "tfidf" "ppmi")
declare -a windowarray=("2" "4" "8" "16")


for r in "${reweightarray[@]}"
do
    for w in "${windowarray[@]}"
    do
        echo ""
        echo "reweight_${r}_window_${w}"
        python train_word_by_word.py --dataset "melisa" --reweight $r --window_size $w --freq_cutoff 5 --vector_dim 300
    done
done