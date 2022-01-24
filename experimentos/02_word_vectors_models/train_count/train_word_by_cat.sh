#! /bin/bash

declare -a reweightarray=("none" "tfidf" "ppmi")
declare -a nclassesarray=("2" "3" "5")

for r in "${reweightarray[@]}"
do
    for n in "${nclassesarray[@]}"
    do
        echo ""
        echo "reweight_${r}"
        python train_word_by_cat.py --dataset "melisa" --max_words 50000 --reweight $r --freq_cutoff 5 --nclasses $n
    done
done