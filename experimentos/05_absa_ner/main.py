from posixpath import split
from utils import load_semeval

def main():
    dataset = load_semeval(split='test')
    char_vocab = dataset['sentence'].str.findall('.').explode().value_counts()
    print(char_vocab.index.tolist())



if __name__ == "__main__":
    main()