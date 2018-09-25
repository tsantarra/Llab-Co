from itertools import product
from collections import OrderedDict


def generate_args():
    vals = OrderedDict([
                        ('process_no', [10]),
                        ('scenario_id', [1]),
                        ('heuristic_id', [1]),
                        ('comm_branch_factor', [3]),
                        ('comm_iterations', [100, 250, 1000]),
                        ('comm_cost', [0]),
                        ('plan_iterations', [100, 500, 1000]),
                        ('experience', [0, 100, 1000]),
                        ('trials', [50]),
                        ])

    for permutation in product(*vals.values()):
        print(' '.join(str(v) for v in permutation))


if __name__ == '__main__':
    generate_args()