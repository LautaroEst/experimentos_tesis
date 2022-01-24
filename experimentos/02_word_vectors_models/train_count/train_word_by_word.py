import os
from utils import *
from models import word_by_word, tfidf_reweight, ppmi_reweight
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

RESULTS_DIR = os.path.join(os.getcwd(),"../../../pretrained_models/")


def train_word_by_word(reader,save_dir,reweight="tfidf",w=2,freq_cutoff=1,max_words=10000,dim=300):
    print("Training word-by-word matrix...")
    X, vocab = word_by_word(reader,w=w,freq_cutoff=freq_cutoff,max_words=max_words)
    print("Total words found:",len(vocab))

    print("Reweighting with {}...".format(reweight))
    if reweight == "none":
        pass
    elif reweight == "tfidf":
        X = tfidf_reweight(X)
    elif reweight == "ppmi":
        X = ppmi_reweight(X)
    else:
        raise TypeError("Reweighting method not supported")

    print("Reducing dimention...")
    pca = TruncatedSVD(n_components=dim)
    X_red = pca.fit_transform(X)
    X_red = (X_red - X_red.mean(axis=0,keepdims=True)) / X_red.std(axis=0).max()

    print("Saving results to word2vec file...")
    with open(os.path.join(save_dir,"word_by_word_{}_w{}_fc{}_dim{}.vec".format(reweight,w,freq_cutoff,dim)),"w") as f:
         f.write("{} {}\n".format(len(vocab),dim))
         for tk, row in tqdm(zip(vocab.keys(),X_red),total=len(vocab)):
            f.write("{} ".format(tk))
            f.write(" ".join(["{:.6f}".format(idx) for idx in row]))
            f.write("\n")


def main():
    args = parse_args_word_by_word()
    if args["dataset"] == "melisa":
        reader = melisa_unsup_reader
    elif args["dataset"] == "sbwc":
        reader = sbwc_reader
    elif args["dataset"] == "melisa+sbwc":
        reader = melisa_sbwc_reader

    train_word_by_word(
        reader=reader,
        save_dir=RESULTS_DIR,
        reweight=args["reweight"],
        w=args["window_size"],
        freq_cutoff=args["freq_cutoff"],
        max_words=args["max_words"],
        dim=args["vector_dim"]
    )


if __name__ == "__main__":
    main()