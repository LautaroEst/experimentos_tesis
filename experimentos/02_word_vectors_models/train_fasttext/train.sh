

# Para entrenar usando vectores preentrenados
~/fastText/fasttext skipgram -input MelisaData.txt -output ../../../pretrained_models/fasttext-sbwc-melisa -pretrainedVectors ../../../pretrained_models/fasttext-sbwc.vec -minCount 1 -dim 300