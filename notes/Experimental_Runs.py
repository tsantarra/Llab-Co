data_config = {}

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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [0, 1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [0, 1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
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
    ('alpha', [1])
]


# 7X - still use a comm heuristic of 0; reduce number of comm iterations, however. 100 is far too many.

# 74 (r74) -  Vary comm branch factor with a few heuristics; BUDGET = BRANCH * ITERATIONS
data_config[74] = [
    ('process_no', [74]),
    ('scenario_id', [2]),
    ('heuristic_id', [3, 4, 10, 11]),
    ('comm_branch_factor', [1, 2, 3, 5]),
    ('comm_iterations', [1, 5, 10, 15, 20]),
    ('comm_cost', [5]),
    ('plan_iterations', [500]),
    ('experience', [100]),
    ('trials', [50]),
    ('alpha', [1])
]


