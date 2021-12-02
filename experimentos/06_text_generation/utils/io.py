import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--devsize', type=float)
    group.add_argument('--dev', action='store_true')
    group.add_argument('--test', action='store_true')
    parser.add_argument('--model',type=str,required=True)
    parser.add_argument('--dataset',type=str,required=True)
    parser.add_argument('--eval_every',type=int,required=True)

    args = vars(parser.parse_args())

    dataset = args['dataset']

    dev, test = args['dev'], args['test']
    if dev and not test:
        split = 'dev'
        devsize = 0.
    elif test and not dev:
        split = 'test'
        devsize = 0.
    else:
        split = 'dev'
        devsize = args['devsize']

    dataset_args = dict(
        dataset=dataset,
        split=split,
        devsize=devsize
    )

    model_name = args['model']
    model_args = dict()

    eval_every = args['eval_every']

    return dataset_args, model_name, model_args, eval_every