# Experimento: Speaker-Listener

En este experimento el objetivo es entrenar un sistema con alguna capacidad de codificar y de generar lenguaje de una manera razonable (en el sentido de "grounding").

Posibles pruebas:

* Entrenar un encoder-decoder que lea un review y genere el título del mismo.

* Entrenar un chat-bot

* Alguna prueba más contextual como la de generar una elección de un comentario a partir de un conjunto de ellos. Otra posibilidad es dar un conjunto de comentarios en forma "ascendente" (1 estrella, 3 estrellas, 5 estrellas) o "descendiente" (1 estrella, 3 estrellas, 5 estrellas)

* Podemos hacer un listener que reciba un comentario y clasifique y un speaker que recibe un título y genera un comentario. O variando el comentario con el título. Lo bueno de esto es que se puede comparar el accuracy de clasificación del listener con otras tareas más simples. Lo malo es que no está muy contextualizada la tarea, en el sentido de que cada comentario es un mundo aparte. Igual se pueden probar variantes contextualizadas. 