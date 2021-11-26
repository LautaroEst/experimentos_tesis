#! /bin/bash

declare -a cls=(2 3 5)

for n in "${cls[@]}" 
do
    echo ""
    echo "LSTM. nclasses: $n"
    echo ""
    python main.py --nclasses $n --lang es --test --max_tokens 60000 --frequency_cutoff 1 --max_sent_len 512 --embedding_dim 300 --rnn "LSTM" --bidirectional --hidden_size 200 --num_layers 1 --dropout 0.2 --num_epochs 8 --batch_size 256 --learning_rate 5e-4 --device "cuda:1" --eval_every 100
    echo ""
    echo "GRU. nclasses: $n"
    echo ""
    python main.py --nclasses $n --lang es --test --max_tokens 60000 --frequency_cutoff 1 --max_sent_len 512 --embedding_dim 300 --rnn "GRU" --hidden_size 200 --num_layers 1 --dropout 0.2 --num_epochs 8 --batch_size 256 --learning_rate 5e-4 --device "cuda:1" --eval_every 100
    echo ""
done

echo ""
echo "RNN. Test."
echo ""
python main.py --nclasses 3 --lang es --test --max_tokens 60000 --frequency_cutoff 1 --max_sent_len 512 --embedding_dim 300 --rnn "RNNrelu" --bidirectional --hidden_size 200 --num_layers 1 --dropout 0.7 --num_epochs 12 --batch_size 256 --learning_rate 5e-4 --device "cuda:1" --eval_every 100
echo ""

echo ""
echo "RNN. Test."
echo ""
python main.py --nclasses 5 --lang es --test --max_tokens 60000 --frequency_cutoff 1 --max_sent_len 512 --embedding_dim 300 --rnn "RNNrelu" --bidirectional --hidden_size 200 --num_layers 1 --dropout 0.7 --num_epochs 12 --batch_size 256 --learning_rate 5e-4 --device "cuda:1" --eval_every 100
echo ""