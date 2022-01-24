from utils import make_tensorboard_visualization, name2file
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
import os

def main():
    with open("words_to_visualize.txt") as f:
        words_to_visualize = [line.split("\n")[0] for line in f.readlines()]

    for name, file in name2file.items():
        make_tensorboard_visualization(name,file,words_to_visualize)


if __name__ == "__main__":
    main()