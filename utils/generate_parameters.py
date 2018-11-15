from itertools import product
from collections import OrderedDict


def generate_args():
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

    for permutation in product(*vals.values()):
        print(' '.join(str(v) for v in permutation))


if __name__ == '__main__':
    generate_args()
