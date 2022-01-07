import os
from utils import *
from models import word_by_cat, tfidf_reweight, ppmi_reweight
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from scipy.sparse import csr_matrix

RESULTS_DIR = os.path.join(os.getcwd(),"../../../pretrained_models/")


def train_word_by_cat(reader,save_dir,reweight="tfidf",freq_cutoff=1,nclasses=5):
    print("Training word-by-cat matrix...")
    X, vocab = word_by_cat(reader,freq_cutoff=freq_cutoff,nclasses=nclasses)
    print("Total words found:",len(vocab))

    print("Reweighting with {}...".format(reweight))
    if reweight == "none":
        pass
    elif reweight == "tfidf":
        X = tfidf_reweight(csr_matrix(X)).toarray()
    elif reweight == "ppmi":
        X = ppmi_reweight(csr_matrix(X)).toarray()
    else:
        raise TypeError("Reweighting method not supported")

    print("Saving results to word2vec file...")
    with open(os.path.join(save_dir,"word_by_cat_{}_fc{}_n{}.vec".format(reweight,freq_cutoff,nclasses)),"w") as f:
         f.write("{} {}\n".format(len(vocab),nclasses))
         for tk, row in tqdm(zip(vocab.keys(),X),total=len(vocab)):
            f.write("{} ".format(tk))
            f.write(" ".join(["{:.6f}".format(idx) for idx in row]))
            f.write("\n")

def main():
    args = parse_args_word_by_cat()
    if args["dataset"] == "melisa":
        reader = melisa_sup_reader
    elif args["dataset"] == "sbwc":
        reader = sbwc_reader
    elif args["dataset"] == "melisa+sbwc":
        reader = melisa_sbwc_reader

    train_word_by_cat(
        reader=reader,
        save_dir=RESULTS_DIR,
        reweight=args["reweight"],
        freq_cutoff=args["freq_cutoff"],
        nclasses=args["nclasses"]
    )


if __name__ == "__main__":
    main()

