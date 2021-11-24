#! /bin/bash

declare -a arr=(10000 20000 40000 60000 80000)

for mf in "${arr[@]}" 
do
    echo "max features: $mf"
    echo ""
    python main.py --nclasses 5 --lang es --devsize 0.05 --ngram_range 1 1 --max_features $mf 
    echo ""
done
