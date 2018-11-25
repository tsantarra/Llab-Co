from itertools import product
from os.path import isfile


def check_logs(vals, log_found=False):

    for permutation in product(*list(val[1] for val in vals)):
        filename = 'data-' + '-'.join(map(str, permutation)) + '.log'

        if not isfile(f'../login.osgconnect.net/out/{filename}'):
            print(' '.join(map(str, permutation)))
        elif log_found:
            print(' '.join(map(str, permutation)) + '\tFound')


if __name__ == '__main__':
    from notes.Experimental_Runs import data_config

    print('Missing: ')
    for experiment_number, parameters in data_config.items():
        if experiment_number < 40:
            continue

        check_logs(parameters, log_found=True)
