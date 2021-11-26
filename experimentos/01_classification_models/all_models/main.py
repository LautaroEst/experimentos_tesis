from models import *

def clf_iterator(nclasses):
    classifiers = {

        NaiveBayesClassifier: dict(
            nclasses=nclasses,
            ngram_range=(1,2),
            max_features=100000
        ),

        FeaturesClassifier: dict(
            nclasses=nclasses,
            ngram_range=(1,3),
            max_features=100000,
            hidden_size=50,
            num_epochs=8,
            batch_size=256,
            learning_rate=5e-4,
            weight_decay=0.0,
            device='cuda:1'
        ),

        CBOWClassifier: dict(
            nclasses=nclasses,
            frequency_cutoff=1,
            max_tokens=60000,
            max_sent_len=512,
            embedding_dim=300,
            hidden_size=200,
            num_layers=4,
            dropout=0.0,
            batch_size=256,
            learning_rate=5e-4,
            num_epochs=16,
            device="cuda:1"
        ),

        RNNClassifier: dict(
            nclasses=nclasses,
            frequency_cutoff=1,
            max_tokens=60000,
            max_sent_len=512,
            embedding_dim=300,
            rnn="RNNrelu",
            bidirectional=True,
            hidden_size=200,
            num_layers=1,
            dropout=0.0,
            batch_size=256,
            learning_rate=5e-4,
            num_epochs=16,
            device="cuda:1"
        ),

        RNNClassifier: dict(
            nclasses=nclasses,
            frequency_cutoff=1,
            max_tokens=60000,
            max_sent_len=512,
            embedding_dim=300,
            rnn="LSTM",
            bidirectional=True,
            hidden_size=200,
            num_layers=1,
            dropout=0.0,
            batch_size=256,
            learning_rate=5e-4,
            num_epochs=16,
            device="cuda:1"
        ),

        RNNClassifier: dict(
            nclasses=nclasses,
            frequency_cutoff=1,
            max_tokens=60000,
            max_sent_len=512,
            embedding_dim=300,
            rnn="GRU",
            bidirectional=False,
            hidden_size=200,
            num_layers=1,
            dropout=0.0,
            batch_size=256,
            learning_rate=5e-4,
            num_epochs=16,
            device="cuda:1"
        ),

        CNNClassifier: dict(
            nclasses=nclasses,
            frequency_cutoff=1,
            max_tokens=60000,
            max_sent_len=512,
            embedding_dim=300,
            n_filters=8,
            filter_sizes=(3,5,7),
            dropout=0.0,
            batch_size=256,
            learning_rate=1e-3,
            num_epochs=1,
            device="cuda:1"
        )
    }
    
    for clf, kwargs in classifiers.items():
        yield clf, kwargs


def main():
    for clf_class, kwargs in clf_iterator(5):
        clf = clf_class(**kwargs)




if __name__ == '__main__':
    main()