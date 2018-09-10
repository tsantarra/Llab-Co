from itertools import product
from collections import OrderedDict


def generate_args():
    vals = OrderedDict([('process_no', [9]),
                        ('scenario_id', [1]),
                        ('heuristic_id', [1]),
                        ('comm_branch_factor', [3, 5]),
                        ('comm_iterations', [100]),
                        ('comm_cost', [0, 2, 5]),
                        ('plan_iterations', [100, 200]),
                        ('experience', [0, 10, 50]),
                        ('trials', [50]),
                        ])

    for permutation in product(*vals.values()):
        print(' '.join(str(v) for v in permutation))


if __name__ == '__main__':
    generate_args()