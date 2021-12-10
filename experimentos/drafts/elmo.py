from elmoformanylangs import Embedder
import os

path_to_model = os.path.join(os.getcwd(),"../../pretrained_models")

def main():
    e = Embedder(path_to_model)

    sents = [["la", "sal", "de", "mar"], ["sal", "de", "ahí"]]
    # the list of lists which store the sentences 
    # after segment if necessary.

    print(e.sents2elmo(sents))
    # will return a list of numpy arrays 
    # each with the shape=(seq_len, embedding_size)

if __name__ == "__main__":
    main()



