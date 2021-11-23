# Observaciones de las diferentes tokenizaciones

Primero se seteó la frecuencia mínima a 1 y se fue agrandando el número de palabras en el vocabulario. Con esos parámetros se midió la cantidad de UNKs en dev.

* Sin procesar y tokenizando por palabras con un patrón "\w+":

| vocab size | min_freq | UNKs per sent |
|------------|----------|---------------|
| 10000      | 60       | 4.76%         |
| 20000      | 18       | 2.59%         |
| 40000      | 5        | 1.41%         |
| 60000      | 2        | 1.04%         |
| 80000      | 1        | 0.87%         |
| 100000     | 1        | 0.80%         |
| 120000     | 1        | 0.75%         |
| 140000     | 1        | 0.68%         |
| 157640     | 1        | 0.63%         |

* Pasando a lower y procesando los acentos, por palabras con un patrón "\w+":

| vocab size | min_freq | UNKs per sent |
|------------|----------|---------------|
| 10000      | 55       | 4.13%         |
| 20000      | 16       | 2.20%         |
| 40000      | 4        | 1.20%         |
| 60000      | 2        | 0.89%         |
| 80000      | 1        | 0.76%         |
| 100000     | 1        | 0.69%         |
| 120000     | 1        | 0.64%         |
| 140000     | 1        | 0.57%         |
| 143897     | 1        | 0.56%         |


* Sin procesar y tokenizando por palabras con un patrón `(\w+|[\.,!\(\)"\-:\?/%;¡\$'¿\\]|\d+)`:

| vocab size | min_freq | UNKs per sent |
|------------|----------|---------------|
| 10000      | 60       | 4.15%         |
| 20000      | 18       | 2.25%         |
| 40000      | 5        | 1.22%         |
| 60000      | 2        | 0.90%         |
| 80000      | 1        | 0.75%         |
| 100000     | 1        | 0.70%         |
| 120000     | 1        | 0.64%         |
| 140000     | 1        | 0.59%         |
| 157657     | 1        | 0.54%         |

* Pasando a lower y procesando los acentos, por palabras con un patrón `(\w+|[\.,!\(\)"\-:\?/%;¡\$'¿\\]|\d+)`:

| vocab size | min_freq | UNKs per sent |
|------------|----------|---------------|
| 10000      | 55       | 3.62%         |
| 20000      | 16       | 1.92%         |
| 40000      | 4        | 1.04%         |
| 60000      | 2        | 0.77%         |
| 80000      | 1        | 0.66%         |
| 100000     | 1        | 0.60%         |
| 120000     | 1        | 0.55%         |
| 140000     | 1        | 0.50%         |
| 143914     | 1        | 0.49%         |

* Sin procesar y tokenizando por palabras con word_tokenize de nltk:

| vocab size | min_freq | UNKs per sent |
|------------|----------|---------------|
| 10000      | 60       | 4.24%         |
| 20000      | 18       | 2.33%         |
| 40000      | 5        | 1.29%         |
| 60000      | 2        | 0.95%         |
| 80000      | 1        | 0.80%         |
| 100000     | 1        | 0.75%         |
| 120000     | 1        | 0.70%         |
| 140000     | 1        | 0.65%         |
| 160000     | 1        | 0.59%         |
| 166160     | 1        | 0.58%         |

* Pasando a lower y procesando los acentos, por palabras con word_tokenize de nltk:

| vocab size | min_freq | UNKs per sent |
|------------|----------|---------------|
| 10000      | 55       | 3.71%         |
| 20000      | 16       | 2.00%         |
| 40000      | 4        | 1.11%         |
| 60000      | 2        | 0.95%         |
| 80000      | 1        | 0.80%         |
| 100000     | 1        | 0.66%         |
| 120000     | 1        | 0.61%         |
| 140000     | 1        | 0.56%         |
| 152653     | 1        | 0.53%         |


