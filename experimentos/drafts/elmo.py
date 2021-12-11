from elmoformanylangs import Embedder
import os
import torch

path_to_model = os.path.join(os.getcwd(),"../../pretrained_models/elmo")

def main():
    e = Embedder(path_to_model)

    sents = [["la", "sal", "de", "mar"], ["sal", "de", "ahí"]]
    # the list of lists which store the sentences 
    # after segment if necessary.

    outs = e.sents2elmo(sents,output_layer=-1)
    max_len = max([o.shape[0] for o in outs])
    batch = torch.zeros(2,max_len,1024,dtype=torch.float)
    for i, o in enumerate(outs):
        batch[i,:o.shape[0],:] = torch.from_numpy(o)
    print(batch)
    # will return a list of numpy arrays 
    # each with the shape=(seq_len, embedding_size)

if __name__ == "__main__":
    main()



