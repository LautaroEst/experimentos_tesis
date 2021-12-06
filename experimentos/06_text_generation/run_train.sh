#! /bin/bash

declare -a dropoutarray=(00 02 05 07)

for dp in "${dropoutarray[@]}" 
do
    echo "Dropout: $dp"
    echo ""
    python main.py --config "config$dp.json" --results_dir "results"
    echo ""
done


