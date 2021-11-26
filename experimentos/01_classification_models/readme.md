# Experimento 1: Algoritmos de clasificación desde cero

En este experimento se implentan algunos algoritmos para clasificar sentimientos. Los algoritmos son entrenados utilizando diferentes datasets, cada uno de los cuales tiene train y test (en principio, para 5, 3 o 2 clases). El entrenamiento se hace con las muestras de train de un dataset y se evaluan usando el test de ese mismo dataset. 

En principio, queremos ver cómo se comportan los diferentes modelos en train y en test y tener un baseline para comparar con los experimentos que siguen.

Las tareas de clasificación implementadas en este punto son las siguientes:

* Clasificación para 2, 3 y 5 categorías con MeLiSA
* Clasificación para 2, 3 y 5 categorías con Amazon
* Clasificación para 2, 3 y 5 categorías con Muchocine
* Clasificación para 2 y 3 categorias con Tass2019 (todos los países juntos).

Accuracy/MAE

| Modelo     | 2-Melisa   | 2-Amazon | 2-Muchocine | 2-TASS |
|------------|------------|----------|-------------|--------|
| BONgrams   | 89.8/8.3   |          |             |        |
| features   | 79.2/18.0  |          |             |        |
| CBOW       | 91.9/2.3   |          |             |        |
| RNNRelu    | 93.4/1.1   |          |             |        |
| LSTM       | 93.8/0.2   |          |             |        |
| GRU        | 93.8/0.3   |          |             |        |
| CNN        | 92.7/2.49  |          |             |        |
| SelfAttn   |          |          |             |        |


| Modelo     | 3-Melisa   | 3-Amazon | 3-Muchocine | 3-TASS |
|------------|------------|----------|-------------|--------|
| BONgrams   | 74.1/27.5  |          |             |        |
| features   | 64.1/46.7  |          |             |        |
| CBOW       | 76.4/16.1  |          |             |        |
| RNNRelu    | 79.4/10.81 |          |             |        |
| LSTM       | 78.6/3.4   |          |             |        |
| GRU        | 78.6/5.6   |          |             |        |
| CNN        | 76.6/16.74 |          |             |        |
| SelfAttn   |          |          |             |        |



| Modelo     | 5-Melisa   | 5-Amazon   | 5-Muchocine |
|------------|------------|------------|-------------|
| BONgrams   | 55.2/49.8  |            |             |
| features   | 44.3/82.1  |          |             |
| CBOW       | 58.6/36.8  |          |             |
| RNNRelu    | 62.3/28.9  |          |             |
| LSTM       | 60.3/15.2  |          |             |
| GRU        | 61.2/19.2  |          |             |
| CNN        | 57.7/37.96 |          |             |
| SelfAttn   |          |          |             |

Hiperparámetros usados:

* BONgrams melisa-5, melisa-3 y melisa-2:
```
ngram_range=(1,3)
max_features=100000
```

* CBOW melisa-5, melisa-3, melisa-2:
```
frequency_cutoff=1,
max_tokens=60000,
max_sent_len=512,
embedding_dim=300,
hidden_size=200,
num_layers=4,
dropout=0.5,
batch_size=256,
learning_rate=5e-4,
num_epochs=8,
```

* Features melisa-2, melisa-3, melisa-5
```
ngram_range=(1,2),
max_features=50000,
hidden_size=400,
num_epochs=16,
batch_size=512,
learning_rate=1e-4,
weight_decay=0.0,
```

* RNN melisa-2, melisa-3, melisa-5
```
frequency_cutoff=1,
max_tokens=60000,
max_sent_len=512,
embedding_dim=300,
rnn="RNNrelu",
bidirectional=True,
hidden_size=200,
num_layers=1,
dropout=0.7,
batch_size=256,
learning_rate=5e-4,
num_epochs=12,
```

* LSTM melisa-2, melisa-3, melisa-5
```
frequency_cutoff=1,
max_tokens=60000,
max_sent_len=512,
embedding_dim=300,
rnn="LSTM",
bidirectional=True,
hidden_size=200,
num_layers=1,
dropout=0.2,
batch_size=256,
learning_rate=5e-4,
num_epochs=8,
```

* GRU melisa-2, melisa-3, melisa-5
```
frequency_cutoff=1,
max_tokens=60000,
max_sent_len=512,
embedding_dim=300,
rnn="GRU",
bidirectional=False,
hidden_size=200,
num_layers=1,
dropout=0.2,
batch_size=256,
learning_rate=5e-4,
num_epochs=8,
```