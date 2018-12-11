from itertools import product


def generate_args(vals):
    all_vals = [val[1] for val in vals] + [['$(Cluster)'], ['$(Process)']]
    for permutation in product(*all_vals):
        print(' '.join(str(v) for v in permutation))


if __name__ == '__main__':
    from notes.Experimental_Runs import data_config

    for exp_no, config in data_config.items():
        if exp_no > 70:
            generate_args(config)
