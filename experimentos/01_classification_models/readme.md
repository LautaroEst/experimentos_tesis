# Expermiento `classification_models`

En este experimento se van a diseñar algunos modelos de clasificación de sentimientos sobre el dataset MeLiSA y se van a dar sus resultados para train y test.

## Modelos basados en extracción de features

1. `TfIdf+NB`: Modelo de extracción de features con un clasificador Naive Bayes. Se implementa una tokenización basada en expresiones regulares. Luego se implementa una vectorización basada en tf-idf y finalmente, una clasificación por Naive Bayes. [Acá](./tfidf_nb/results_5classes/2021-11-10-20-31-24_classification_report.log) se muestran los resultados para 5 clases y [acá](./tfidf_nb/results_3classes/2021-11-10-22-03-12_classification_report.log) para 3 clases. Ambos fueron obtenidos con los siguientes hiperparámetros (los mejores obtenidos sobre el conjunto de dev):

- Patrón para tokenizar: `(\w+|[\.,!\(\)"\-:\?/%;¡\$'¿\\]|\d+)`
- Frecuencia de documento mínima: 1
- Rango de los n-gramas: (1, 3)
- Cantidad máxima de features en el vocabulario: 500000

Los resultados para 2 clases se encuentran [acá](./tfidf_nb/results_2classes/2021-11-10-22-29-45_classification_report.log) y los hiperparámetros usados fueron:

- Patrón para tokenizar: (\w+|[!?\.,])
- Frecuencia de documento mínima: 1
- Rango de los n-gramas: (1, 3)
- Cantidad máxima de features en el vocabulario: 100000

2. `CatVectorizer+2LayerNet`: Modelo basado en extracción de features con un clasificador Softmax de dos capas. Se implementa una tokenización basada en expresiones regulares, una normalización estándar y una bolsa de palabras por conteo de categoría con un máximo de 10000 palabras. Luego se implementa una clasificación con una red neuronal de 2 capas.

## Modelos basados en Redes Neuronales

Acá se presentan 3 modelos neuronales end-to-end (uno Feed Forward, uno recurrente y uno convolucional), los cuales serán entrenados de diferentes maneras. Primero, se implementarán sin ningún tipo de información adicional (sin vectores preentrenados). Luego se incorporarán vectores word2vec y glove y se probará entrenar cada modelo haciendo ajuste fino de toda la red (primero) y de todo lo que no es el layer de embedding (después). Finalmente se utilizarán vectores FastText como entrada de los tres modelos, y ya que estos vectores están fijos, los parámetros se entrenan en todas las capas posteriores a las de embeddings.

3. `pretrainedCBOW+FFNet`: Modelo basado en una bolsa de palabras continua con embeddings preentrenados (word2vec, Glove y FastText). La vectorización está dada por los vectores preentrenados y la clasificación está hecha con una bolsa de palabras continua seguida de una red feedforward.

4. `embedding+RNN`: Modelo basado en un layer embedding y una red recurrente GRU. Las pruebas se hacen para embeddings preentrenados y sin preentrenar.

5. `embedding+CNN`: Idem pero con un modelo basado en redes convolucionales con varios filtros.

## Modelos basados en transformers:

Estos modelos son los más fáciles de entrenar porque básicamente el tema del vocabulario ya está resuelto. Queda fijo y listo. 

6. `BETO`: Clasificación con el modelo BETO preentrenado en sus versiones cased y uncased.

# Resultados

* 5 clases en español:

| Modelo | Train accuracy | Test accuracy |
|--------|----------------|---------------|
| 1      |     68%        |      55%      |
| 2      |                |               |
| 3      |                |               |
| 4      |                |               |
| 5      |                |               |
| 6      |                |               |

* 3 clases en español:

| Modelo | Train accuracy | Test accuracy |
|--------|----------------|---------------|
| 1      |      79%       |       73%     |
| 2      |                |               |
| 3      |                |               |
| 4      |                |               |
| 5      |                |               |
| 6      |                |               |

* 2 clases en español:

| Modelo | Train accuracy | Test accuracy |
|--------|----------------|---------------|
| 1      |      92%       |      90%      |
| 2      |                |               |
| 3      |                |               |
| 4      |                |               |
| 5      |                |               |
| 6      |                |               |

