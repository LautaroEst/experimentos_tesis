# Expermiento `run_models`

En este experimento se van a diseñar algunos modelos de clasificación de sentimientos sobre el dataset MeLiSA y se van a dar sus resultados para train, dev y test.

Modelos implementados hasta el momento:

* `TfIdf+NB`: Modelo de extracción de features con un clasificador Naive Bayes. Se implementa una tokenización basada en expresiones regulares, una normalización estándar, y una bolsa de ngramas (1,2) con un máximo de 10000 ngramas. Luego se implementa una vectorización basada en tf-idf y finalmente, una clasificación por Naive Bayes. 

* `CatVectorizer+2LayerNet`: Modelo basado en extracción de features con un clasificador Softmax de dos capas. Se implementa una tokenización basada en expresiones regulares, una normalización estándar y una bolsa de palabras por conteo de categoría con un máximo de 10000 palabras. Luego se implementa una clasificación con una red neuronal de 2 capas.

* `pretrainedCBOW+FFNet`: Modelo basado en una bolsa de palabras continua con embeddings preentrenados (word2vec, Glove y FastText). La vectorización está dada por los vectores preentrenados y la clasificación está hecha con una bolsa de palabras continua seguida de una red feedforward.

* `embedding+RNN`: Modelo basado en un layer embedding y una red RNN. El tipo de activación recurrente utilizada es ... y las pruebas se hacen para embeddings preentrenados y sin preentrenar.

* `embedding+CNN`: Idem pero con CNN

* `BETO`: Clasificación con el modelo BETO preentrenado. 