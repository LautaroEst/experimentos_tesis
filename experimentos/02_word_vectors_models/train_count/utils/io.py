import argparse


def parse_args_word_by_word():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,required=True)
    parser.add_argument('--window_size',type=int,required=True)
    parser.add_argument('--freq_cutoff',type=int,required=True)
    parser.add_argument('--vector_dim',type=int,required=True)
    parser.add_argument('--reweight',type=str,required=True)
    args = vars(parser.parse_args())
    return args


def parse_args_word_by_cat():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',type=str,required=True)
    parser.add_argument('--freq_cutoff',type=int,required=True)
    parser.add_argument('--nclasses',type=int,required=True)
    parser.add_argument('--reweight',type=str,required=True)
    args = vars(parser.parse_args())
    return args