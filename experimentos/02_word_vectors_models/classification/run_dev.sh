#! /bin/bash

declare -a datasetarray=("amazon" "cine" "melisa" "tass")
declare -a wvarray=("none" "word2vec" "glove" "fasttext" "elmo")
declare -a modelarray=("cbow" "lstm" "cnn")
declare -a dropoutarray=("00" "01" "02")
declare -a lrarray=("1e4" "5e4")

maxsegfaults="10"
segfaults="0"

for wv in "${wvarray[@]}"
do
    for d in "${datasetarray[@]}"    
    do
        for m in "${modelarray[@]}"
        do
            for dp in "${dropoutarray[@]}"
            do
                for lr in "${lrarray[@]}"
                do
                    i="0"
                    echo "$i"
                    while [[ $i -lt 3 ]] && [[ $segfaults -lt $maxsegfaults ]]
                    do
                        echo ""
                        echo "${d}_${wv}_${m} (run $i)"
                        python main.py --config "./configslr1e4dp00/${d}_${wv}_${m}.json" --results_dir "./results/${d}_${wv}_${m}/"
                        if [[ $? -eq 139 ]] 
                        then
                            echo "hubo seg fault. $segfaults"
                            segfaults=$[$segfaults+1]
                        else
                            echo "no hubo seg fault"
                            i=$[$i+1]
                        fi
                    done
                    segfaults="0"
                done
            done
        done
    done
done

# python main.py --config "./configs/amazon_none_cbow.json" --results_dir "./results"
# python main.py --config "./configs/melisa_none_cbow.json" --results_dir "./results"
# python main.py --config "./configs/melisa_none_lstm.json" --results_dir "./results"
# python main.py --config "./configs/melisa_none_cnn.json" --results_dir "./results"
# python main.py --config "./configs/tass_none_lstm.json" --results_dir "./results"
# python main.py --config "./configs/tass_none_cnn.json" --results_dir "./results"

# python main.py --config "./configs/amazon_word2vec_lstm.json" --results_dir "./results"
# python main.py --config "./configs/amazon_word2vec_cnn.json" --results_dir "./results"
# python main.py --config "./configs/cine_word2vec_lstm.json" --results_dir "./results"
# python main.py --config "./configs/cine_word2vec_cnn.json" --results_dir "./results"
# python main.py --config "./configs/melisa_word2vec_cnn.json" --results_dir "./results"
# python main.py --config "./configs/melisa_glove_cbow.json" --results_dir "./results"
# python main.py --config "./configs/melisa_glove_lstm.json" --results_dir "./results"
# python main.py --config "./configs/cine_glove_lstm.json" --results_dir "./results"
# python main.py --config "./configs/melisa_elmo_cbow.json" --results_dir "./results"
# python main.py --config "./configs/melisa_elmo_lstm.json" --results_dir "./results"
# python main.py --config "./configs/melisa_elmo_cnn.json" --results_dir "./results"
# python main.py --config "./configs/cine_elmo_cnn.json" --results_dir "./results"
# python main.py --config "./configs/cine_elmo_lstm.json" --results_dir "./results"
# python main.py --config "./configs/cine_elmo_cbow.json" --results_dir "./results"
# python main.py --config "./configs/amazon_elmo_cnn.json" --results_dir "./results"
# python main.py --config "./configs/amazon_elmo_lstm.json" --results_dir "./results"
# python main.py --config "./configs/amazon_elmo_cbow.json" --results_dir "./results"
# python main.py --config "./configs/amazon_fasttext_cbow.json" --results_dir "./results"

