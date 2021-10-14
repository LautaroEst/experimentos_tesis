# Experimento: Entrenamiento de Word Embeddings con MeLiSA

En este experimento vamos a tratar de entrenar vectores de palabras utilizando el dataset MeLiSA. 

Algoritmos utilizados:

* Métodos de conteo (`WordByWord` y `WordByCategory`): métodos en donde los vectores se entrenan contando palabras en sus contextos. Las variantes son `WordByWord`, en donde se cuenta dentro de una ventana y `WordByCategory` en donde se cuenta según la cateogría a la que pertenece (1 estrella, 2 estrellas, etc.)

* `word2vec`: Método básico para entrenar vectores de palabras con una red neuronal. La variante utilizada es Skipgram con negative sampling.

* `GloVe`: Método neuronal también básico pero con incorporación de los métodos de conteo.

* `FastText`: Método neuronal que incorpora el entrenamiento utilizando subpalabras.

* La lista sigue...

Métodos de evaluación:

* ...

