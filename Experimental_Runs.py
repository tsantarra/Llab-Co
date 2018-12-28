from itertools import product
from os import listdir


data_config = {}
baselines = {}

# 9 - test runs
data_config[9] = [
    ('process_no', [9]),
    ('scenario_id', [1]),
    ('heuristic_id', [1]),
    ('comm_branch_factor', [3, 5]),
    ('comm_iterations', [100]),
    ('comm_cost', [0, 2, 5]),
    ('plan_iterations', [100, 200]),
    ('experience', [0, 10, 50]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 10 - test plan size/experience
data_config[10] = [
    ('process_no', [10]),
    ('scenario_id', [1]),
    ('heuristic_id', [1]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100, 250, 1000]),  # comm iterations
    ('comm_cost', [0]),
    ('plan_iterations', [100, 500, 1000]),  # plan iterations
    ('experience', [0, 100, 1000]),  # experience
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 11 - test every new heuristic
data_config[11] = [
    ('process_no', [11]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(12))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 12 - test plan size/experience - NEW MAZE
data_config[12] = [
    ('process_no', [12]),
    ('scenario_id', [1]),
    ('heuristic_id', [1]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100, 250, 500]),  # comm iterations
    ('comm_cost', [1]),
    ('plan_iterations', [100, 500, 1000]),  # plan iterations
    ('experience', [0, 100, 1000]),  # experience
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 13 - test every new heuristic - NEW MAZE
data_config[13] = [
    ('process_no', [13]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(12))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 14 - test heuristic baselines (random, next most likely state) - NEW MAZE
data_config[14] = [
    ('process_no', [14]),
    ('scenario_id', [1]),
    ('heuristic_id', [12, 13]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 15 - test other 'baseline' ---> no comms
data_config[15] = [
    ('process_no', [15]),
    ('scenario_id', [1]),
    ('heuristic_id', [0]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [1000]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 16 - retest all heuristics, including baselines
data_config[16] = [
    ('process_no', [13]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 17 - Rerun 15 and 16 with action value fixes, optimizations, and heuristic
data_config[17] = [
    ('process_no', [17]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 18 - Test all heuristics on a.maze (12 rounds)
data_config[18] = [
    ('process_no', [18]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 19 - Test all heuristics on a.maze (12 rounds), CRP alpha=0
data_config[19] = [
    ('process_no', [19]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 23 - Run greedy heuristics
data_config[23] = [
    ('process_no', [23]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [100000]),
    ('comm_iterations', [1]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 24 -  Test comm branch factor with a few heuristics
data_config[24] = [
    ('process_no', [24]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3, 5, 10, 100]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 25 - Test comm costs with a few heuristics
data_config[25] = [
    ('process_no', [25]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0, 1, 5, 10]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 26 - Experience with a few heuristics
data_config[26] = [
    ('process_no', [26]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 25, 100, 500]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 32 - Rerun all base heuristics on big maze. Alpha 0/1
data_config[32] = [
    ('process_no', [32]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0, 1]),
    ('policy_cap', [0]),
]

# 33 - Rerun all heuristics, but with only 1 comm planning iteration
data_config[33] = [
    ('process_no', [33]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [100000]),
    ('comm_iterations', [1]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 34 (r24) -  Test comm branch factor with a few heuristics
data_config[34] = [
    ('process_no', [34]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3, 5, 10, 100]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 35 (r25) - Test comm costs with a few heuristics
data_config[35] = [
    ('process_no', [35]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0, 1, 5, 10]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 36 (r26) - Experience with a few heuristics
data_config[36] = [
    ('process_no', [36]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 25, 100, 500]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 42 - Rerun all base heuristics on big maze. Alpha 0/1
data_config[42] = [
    ('process_no', [42]),
    ('scenario_id', [1]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0, 1]),
    ('policy_cap', [0]),
]

# 43 - Rerun all heuristics, but with only 1 comm planning iteration
data_config[43] = [
    ('process_no', [43]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [300]),
    ('comm_iterations', [1]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 44 (r34) -  Test comm branch factor with a few heuristics
data_config[44] = [
    ('process_no', [44]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [1, 3, 5, 10, 100]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 45 (r35) - Test comm costs with a few heuristics
data_config[45] = [
    ('process_no', [45]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0, 1, 5, 10]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 46 (r36) - Experience with a few heuristics
data_config[46] = [
    ('process_no', [46]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 25, 100, 500]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]


# 5X - Comm cost > 0 now, as we continue communicating until no benefit (all policy for cost = 0)
# log numbers 1146804

# 53 - Rerun all heuristics, but with only 1 comm planning iteration
data_config[53] = [
    ('process_no', [53]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [500]),
    ('comm_iterations', [1]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 54 (r34) -  Vary comm branch factor with a few heuristics; BUDGET = BRANCH * ITERATIONS
data_config[54] = [
    ('process_no', [54]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3, 5, 10]),
    ('comm_iterations', [10, 60, 100]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 55 (r35) - Vary comm costs with a few heuristics
data_config[55] = [
    ('process_no', [55]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [1, 10, 99]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 56 (r36) - Vary experience with a few heuristics
data_config[56] = [
    ('process_no', [56]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 25, 100, 500]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 6X - Comm heuristic = 0

# 63 - Rerun all heuristics, but with only 1 comm planning iteration
data_config[63] = [
    ('process_no', [63]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [500]),
    ('comm_iterations', [1]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 64 (r34) -  Vary comm branch factor with a few heuristics; BUDGET = BRANCH * ITERATIONS
data_config[64] = [
    ('process_no', [64]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3, 5, 10]),
    ('comm_iterations', [10, 60, 100]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 65 (r35) - Vary comm costs with a few heuristics
data_config[65] = [
    ('process_no', [65]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [1, 10, 99]),
    ('plan_iterations', [600]),
    ('experience', [100]),
    ('trials', [60]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 66 (r36) - Vary experience with a few heuristics
data_config[66] = [
    ('process_no', [66]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [100]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 25, 100, 500]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]


# 7X - still use a comm heuristic of 0; reduce number of comm iterations, however. 100 is far too many.

# 74 (r74) -  Vary comm branch factor with all heuristics; BUDGET = BRANCH * ITERATIONS
data_config[74] = [
    ('process_no', [74]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [1, 2, 3, 5]),
    ('comm_iterations', [1, 5, 10, 15, 20]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 75 -  Vary experience with all heuristics. Certain heuristics will (hopefully) work better with less experience.
data_config[75] = [
    ('process_no', [75]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [20]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 100, 1000]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 76 -  Vary cost
data_config[76] = [
    ('process_no', [76]),
    ('scenario_id', [2]),
    ('heuristic_id', [0, 4, 11, 12]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [20]),
    ('comm_cost', [0, 5, 25, 50, 99]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 8X series - Added policy cap parameters (caps the number of unique teammate policies)
# 81 -  Vary experience with policy cap
data_config[81] = [
    ('process_no', [81]),
    ('scenario_id', [2]),
    ('heuristic_id', [0, 4, 11, 12]),
    ('comm_branch_factor', [3]),
    ('comm_iterations', [20]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [10, 100, 1000]),
    ('trials', [100]),
    ('alpha', [1]),
    ('policy_cap', [0, 5, 25, 125])
]

# 84 -  Vary comm branch factor with all heuristics; BUDGET = BRANCH * ITERATIONS
data_config[84] = [
    ('process_no', [84]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [1, 3, 5]),
    ('comm_iterations', [1, 10, 20]),
    ('comm_cost', [1]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [100]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]

# 85 - COST 1. Vary experience with all heuristics.
data_config[85] = [
    ('process_no', [85]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [5]),
    ('comm_iterations', [20]),
    ('comm_cost', [1]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 100, 1000]),
    ('trials', [100]),
    ('alpha', [1]),
    ('policy_cap', [0]),
]


# 9X - Used MLE alpha for CRP

# 91 -  Vary experience with policy cap
data_config[91] = [
    ('process_no', [91]),
    ('scenario_id', [2]),
    ('heuristic_id', [0, 4, 11, 12]),       # Find top heuristics befre
    ('comm_branch_factor', [5]),
    ('comm_iterations', [20]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [10, 100, 1000]),
    ('trials', [100]),
    ('alpha', [0]),
    ('policy_cap', [0, 5, 25, 125])
]

# 94 - Vary search params
data_config[94] = [
    ('process_no', [94]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [1, 3, 5]),
    ('comm_iterations', [1, 10, 20]),
    ('comm_cost', [1]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]

# 95 - Vary experience with all heuristics.
data_config[95] = [
    ('process_no', [95]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [5]),
    ('comm_iterations', [20]),
    ('comm_cost', [1]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 100, 1000]),
    ('trials', [100]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]

# 96 -  Vary cost
data_config[96] = [
    ('process_no', [96]),
    ('scenario_id', [2]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [5]),
    ('comm_iterations', [20]),
    ('comm_cost', [1, 5, 10, 99]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]


# 10X - small2 maze. Fixed alpha. Fixed C&R Heuristic.

# 101 - Vary search params
data_config[101] = [
    ('process_no', [101]),
    ('scenario_id', [6]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [1, 3, 5]),
    ('comm_iterations', [1, 10, 20]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]
baselines[101] = [
    ('process_no', [101]),
    ('scenario_id', [6]),
    ('heuristic_id', [0]),
    ('comm_branch_factor', [0]),
    ('comm_iterations', [0]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]

# 102 -  Vary cost
data_config[102] = [
    ('process_no', [102]),
    ('scenario_id', [6]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [5]),
    ('comm_iterations', [10]),
    ('comm_cost', [1, 5, 10, 99]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]
baselines[102] = [
    ('process_no', [102]),
    ('scenario_id', [6]),
    ('heuristic_id', [0]),
    ('comm_branch_factor', [0]),
    ('comm_iterations', [0]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]

# 103 - Vary experience with all heuristics.
data_config[103] = [
    ('process_no', [103]),
    ('scenario_id', [6]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [5]),
    ('comm_iterations', [10]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 100, 1000]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]
baselines[103] = [
    ('process_no', [103]),
    ('scenario_id', [6]),
    ('heuristic_id', [0]),
    ('comm_branch_factor', [0]),
    ('comm_iterations', [0]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 100, 1000]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]

# 104 -  Vary experience with policy cap  --------------------- TODO
data_config[104] = [
    ('process_no', [104]),
    ('scenario_id', [6]),
    ('heuristic_id', [0, 4, 11, 12]),       # Find top heuristics befre
    ('comm_branch_factor', [5]),
    ('comm_iterations', [20]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [10, 100, 1000]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0, 5, 25, 125])
]

# 105 - Domain structure
data_config[105] = [
    ('process_no', [105]),
    ('scenario_id', [7]),
    ('heuristic_id', list(range(14))),
    ('comm_branch_factor', [5]),
    ('comm_iterations', [10]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 100, 1000]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]
baselines[105] = [
    ('process_no', [105]),
    ('scenario_id', [7]),
    ('heuristic_id', [0]),
    ('comm_branch_factor', [0]),
    ('comm_iterations', [0]),
    ('comm_cost', [0]),
    ('plan_iterations', [500]),
    ('experience', [0, 10, 100, 1000]),
    ('trials', [50]),
    ('alpha', [0]),
    ('policy_cap', [0]),
]


def generate_args(trial_no):
    if trial_no in baselines:
        vals = baselines[trial_no]
        all_vals = [val[1] for val in vals] + [['$(Cluster)'], ['$(Process)']]
        for permutation in product(*all_vals):
            print(' '.join(str(v) for v in permutation))

    vals = data_config[trial_no]
    all_vals = [val[1] for val in vals] + [['$(Cluster)'], ['$(Process)']]
    for permutation in product(*all_vals):
        print(' '.join(str(v) for v in permutation))


def check_logs(trial_no, directory, log_found=False, min_count=1):
    if trial_no in baselines:
        vals = baselines[trial_no]
        filenames = listdir(directory)
        for permutation in product(*list(val[1] for val in vals)):
            file_start = 'data-' + '-'.join(map(str, permutation))
            permutation_count = sum(1 for filename in filenames if filename.startswith(file_start))

            if permutation_count < min_count:
                print(' '.join(map(str, permutation)) + ' $(Cluster) $(Process)')
            elif log_found:
                print(' '.join(map(str, permutation)) + '\tFound ' + str(permutation_count))

    vals = data_config[trial_no]
    filenames = listdir(directory)
    for permutation in product(*list(val[1] for val in vals)):
        file_start = 'data-' + '-'.join(map(str, permutation))
        permutation_count = sum(1 for filename in filenames if filename.startswith(file_start))

        if permutation_count < min_count:
            print(' '.join(map(str, permutation)) + ' $(Cluster) $(Process)')
        elif log_found:
            print(' '.join(map(str, permutation)) + '\tFound ' + str(permutation_count))


if __name__ == '__main__':
    #generate_args(105)
    check_logs(94, 'login.osgconnect.net/out/')
    check_logs(96, 'login.osgconnect.net/out/')
    check_logs(101, 'login.osgconnect.net/out/')
    check_logs(102, 'login.osgconnect.net/out/')
    check_logs(105, 'login.osgconnect.net/out/')