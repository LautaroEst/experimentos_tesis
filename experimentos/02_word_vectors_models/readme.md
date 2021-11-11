# Experimento: Entrenamiento de Word Embeddings con MeLiSA

En este experimento vamos a tratar de entrenar vectores de palabras utilizando el dataset MeLiSA. 

## Algoritmos utilizados:

* Métodos de conteo (`WordByWord` y `WordByCategory`): métodos en donde los vectores se entrenan contando palabras en sus contextos. Las variantes son `WordByWord`, en donde se cuenta dentro de una ventana y `WordByCategory` en donde se cuenta según la cateogría a la que pertenece (1 estrella, 2 estrellas, etc.)

* `word2vec`: Método básico para entrenar vectores de palabras con una red neuronal. La variante utilizada es Skipgram con negative sampling.

* `GloVe`: Método neuronal también básico pero con incorporación de los métodos de conteo.

* `FastText`: Método neuronal que incorpora el entrenamiento utilizando subpalabras.

* `ELMO`?

* `BERT`


## Métodos de evaluación:

Los métodos para evaluar Word Embeddings se dividen en dos tipos: intrínsecos, que incluyen tareas como analogía, similitud, detección de outlier y otros en donde los vectores obtenidos se evalúan entre sí con algún criterio; y extrínsecos, que consisten en evaluar el desempeño de una tarea incorporando word embeddings entrenados.

Evaluación intrínseca:

* Buscar datasets en español...

Evaluación extrínseca:

* Clasificación de tweets con el dataset InterTASS

* Clasificación de críticas de películas con el dataset corpusCine

* Análisis de sentimientos basado en aspectos con ABSA2016 (SemEval task 5)


## Análisis de Sesgos en los vectores

Después de hacer todo esto hay que hacer un análisis de los sesgos presentes en las representaciones obtenidas.