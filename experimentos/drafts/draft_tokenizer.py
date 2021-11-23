from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

train_corpus = [
    "low","low","low","low","low","lowest","lowest",
    "newer","newer","newer","newer","newer","newer",
    "wider","wider","wider","new","new"
]
with open('sample.txt','w') as f:
    f.write(' '.join(train_corpus))


def main():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    files = ['sample.txt']
    tokenizer.train(files, trainer)
    tokenizer.save("tokenizer.json")

if __name__ == "__main__":
    main()
    