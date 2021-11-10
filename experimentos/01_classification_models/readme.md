# Expermiento `run_models`

En este experimento se van a diseñar algunos modelos de clasificación de sentimientos sobre el dataset MeLiSA y se van a dar sus resultados para train, dev y test.

Modelos implementados hasta el momento:

1. `TfIdf+NB`: Modelo de extracción de features con un clasificador Naive Bayes. Se implementa una tokenización basada en expresiones regulares, una normalización estándar, y una bolsa de ngramas (1,2) con un máximo de 10000 ngramas. Luego se implementa una vectorización basada en tf-idf y finalmente, una clasificación por Naive Bayes. 

2. `CatVectorizer+2LayerNet`: Modelo basado en extracción de features con un clasificador Softmax de dos capas. Se implementa una tokenización basada en expresiones regulares, una normalización estándar y una bolsa de palabras por conteo de categoría con un máximo de 10000 palabras. Luego se implementa una clasificación con una red neuronal de 2 capas.

3. `pretrainedCBOW+FFNet`: Modelo basado en una bolsa de palabras continua con embeddings preentrenados (word2vec, Glove y FastText). La vectorización está dada por los vectores preentrenados y la clasificación está hecha con una bolsa de palabras continua seguida de una red feedforward.

4. `embedding+RNN`: Modelo basado en un layer embedding y una red recurrente GRU. Las pruebas se hacen para embeddings preentrenados y sin preentrenar.

5. `embedding+CNN`: Idem pero con un modelo basado en redes convolucionales con varios filtros.

6. `BETO`: Clasificación con el modelo BETO preentrenado en sus versiones cased y uncased.

# Resultados

* 5 clases en español:

| Modelo | Train accuracy | Test accuracy |
|--------|----------------|---------------|
| 1      |                |               |
| 2      |                |               |
| 3      |                |               |
| 4      |                |               |
| 5      |                |               |
| 6      |                |               |

* 3 clases en español:

| Modelo | Train accuracy | Test accuracy |
|--------|----------------|---------------|
| 1      |                |               |
| 2      |                |               |
| 3      |                |               |
| 4      |                |               |
| 5      |                |               |
| 6      |                |               |

* 2 clases en español:

| Modelo | Train accuracy | Test accuracy |
|--------|----------------|---------------|
| 1      |                |               |
| 2      |                |               |
| 3      |                |               |
| 4      |                |               |
| 5      |                |               |
| 6      |                |               |

