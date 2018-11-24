from collections import OrderedDict
from itertools import product
from os.path import isfile


def check_logs():
    vals = OrderedDict([
        ('process_no', [46]),
        ('scenario_id', [2]),
        ('heuristic_id', [3, 4, 10, 11]),
        ('comm_branch_factor', [3]),
        ('comm_iterations', [100]),
        ('comm_cost', [0]),
        ('plan_iterations', [500]),
        ('experience', [0, 10, 25, 100, 500]),
        ('trials', [50]),
        ('alpha', [1])
    ])

    print('Missing: ')
    for permutation in product(*vals.values()):
        filename = 'data-' + '-'.join(map(str, permutation)) + '.log'

        if not isfile(f'../login.osgconnect.net/out/{filename}'):
            print(' '.join(map(str, permutation)))


if __name__ == '__main__':
    check_logs()
