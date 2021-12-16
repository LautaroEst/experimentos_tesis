#! /bin/bash

declare -a datasetarray=("amazon" "cine" "melisa" "tass")
declare -a wvarray=("none" "word2vec" "glove" "fasttext" "elmo")
declare -a modelarray=("cbow" "lstm" "cnn")

mkdir "results"
for wv in "${wvarray[@]}"
do
    for d in "${datasetarray[@]}"    
    do
        for m in "${modelarray[@]}"
        do
            echo ""
            echo "${d}_${wv}_${m}"
            mkdir "results/${d}_${wv}_${m}"
        done
    done
done
