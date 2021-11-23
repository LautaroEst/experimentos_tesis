from itertools import chain
from collections import Counter
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers

train_data = """Parece mentira Carlitos estamos volviendo
Volviendo estamos Carlitos mentira parece
Estamos volviendo después de veinte años veinte largos años
Tenés razón Carlitos fueron veinte años muy largos De trece catorce meses
""".lower().split('\n')

test_data = """Aunque pensándolo bien Carlitos qué son veinte años
Son doscientos cuarenta meses
Gracias Carlitos 
De nada Carlitos""".lower().split('\n')



def bpe():
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    # tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # tokenizer.decoders = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=100000,
        #show_progress=True,
        min_frequency=1,
        special_tokens=["[UNK]"],
    )

    tokenizer.train_from_iterator(train_data, trainer=trainer)
    print(tokenizer.get_vocab().keys())
    for sent in tokenizer.encode_batch(test_data):
        print(sent.tokens)
    # print(tokenizer.encode(test_data[0]).ids)
    # print(tokenizer.token_to_id('not'))

def wordpiece():
    tokenizer = Tokenizer(models.WordPiece(unk_token='[UNK]'))
    # tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # tokenizer.decoders = decoders.ByteLevel()

    trainer = trainers.WordPieceTrainer(
        vocab_size=100000,
        #show_progress=True,
        min_frequency=1,
        special_tokens=["[UNK]"],
    )

    tokenizer.train_from_iterator(train_data, trainer=trainer)
    print(tokenizer.get_vocab().keys())
    for sent in tokenizer.encode_batch(test_data):
        print(sent.tokens)

def unigram():
    tokenizer = Tokenizer(models.Unigram())
    # tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    # tokenizer.decoders = decoders.ByteLevel()

    trainer = trainers.UnigramTrainer(
        unk_token="[UNK]",
        vocab_size=100000,
        #show_progress=True,
        min_frequency=1,
        special_tokens=["[UNK]"]
    )

    tokenizer.train_from_iterator(train_data, trainer=trainer)
    print(tokenizer.get_vocab().keys())
    for sent in tokenizer.encode_batch(test_data):
        print(sent.tokens)
        print(sent.ids)

def character():
    tokenized_train_corpus = [list(c) for sent in train_data for c in sent.split(' ')]
    counts = dict(Counter(chain(*tokenized_train_corpus)))
    vocab = {tk: idx for idx, tk in enumerate(counts.keys(),1)}
    vocab['[UNK]'] = 0
    tokenized_test_corpus = []
    for sent in test_data:
        sent = sent.split(' ')
        tokenized_sent = []
        for tk in sent:
            for char in list(tk):
                if char not in vocab:
                    tokenized_sent.append('[UNK]')
                else:
                    tokenized_sent.append(char)
        tokenized_test_corpus.append(tokenized_sent)
    print(vocab.keys())
    print(tokenized_test_corpus)


def main():
    # bpe()
    # wordpiece()
    unigram()
    # character()


if __name__ == "__main__":
    main()