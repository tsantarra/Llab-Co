from itertools import product
from collections import OrderedDict


def generate_args():
    vals = OrderedDict([
        ('process_no', [15]),
        ('scenario_id', [1]),
        ('heuristic_id', [0]),
        ('comm_branch_factor', [3]),
        ('comm_iterations', [100]),
        ('comm_cost', [1000]),
        ('plan_iterations', [500]),
        ('experience', [100]),
        ('trials', [50]),
                        ])

    for permutation in product(*vals.values()):
        print(' '.join(str(v) for v in permutation))


if __name__ == '__main__':
    generate_args()