
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
data_config[9] = [
('process_no', [10]),
('scenario_id', [1]),
('heuristic_id', [1]),
('comm_branch_factor', [3]),
('comm_iterations', [100, 250, 1000]),  # comm iterations
('comm_cost', [0]),
('plan_iterations', [100, 500, 1000]),  # plan iterations
('experience', [0, 100, 1000]),         # experience
('trials', [50]),
('alpha', [1])
]

# 11 - test every new heuristic
data_config[9] = [
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
data_config[9] = [
('process_no', [12]),
('scenario_id', [1]),
('heuristic_id', [1]),
('comm_branch_factor', [3]),
('comm_iterations', [100, 250, 500]),  # comm iterations
('comm_cost', [1]),
('plan_iterations', [100, 500, 1000]),  # plan iterations
('experience', [0, 100, 1000]),         # experience
('trials', [50]),
('alpha', [1])
]

# 13 - test every new heuristic - NEW MAZE
data_config[9] = [
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
data_config[9] = [
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
data_config[9] = [
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
data_config[9] = [
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
data_config[9] = [
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
data_config[9] = [
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
data_config[9] = [
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
data_config[9] = [
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
data_config[9] = [
('process_no', [24]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3,5,10,100]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [1])
]

# 25 - Test comm costs with a few heuristics
data_config[9] = [
('process_no', [25]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0, 1, 5, 10]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [1])
]

# 26 - Experience with a few heuristics
data_config[9] = [
('process_no', [26]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [0, 10, 25, 100, 500]),
('trials', [50]),
('alpha', [1])
]



# 32 - Rerun all base heuristics on big maze. Alpha 0/1
data_config[9] = [
('process_no', [32]),
('scenario_id', [1]),
('heuristic_id', list(range(14))),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [0,1])
]

# 33 - Rerun all heuristics, but with only 1 comm planning iteration
data_config[9] = [
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
data_config[9] = [
('process_no', [34]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3,5,10,100]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [1])
]

# 35 (r25) - Test comm costs with a few heuristics
data_config[9] = [
('process_no', [35]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0, 1, 5, 10]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [1])
]

# 36 (r26) - Experience with a few heuristics
data_config[9] = [
('process_no', [36]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [0, 10, 25, 100, 500]),
('trials', [50]),
('alpha', [1])
]




# 42 - Rerun all base heuristics on big maze. Alpha 0/1
data_config[9] = [
('process_no', [42]),
('scenario_id', [1]),
('heuristic_id', list(range(14))),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [0,1])
]

# 43 - Rerun all heuristics, but with only 1 comm planning iteration
data_config[9] = [
('process_no', [43]),
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


# 44 (r34) -  Test comm branch factor with a few heuristics
data_config[9] = [
('process_no', [44]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3,5,10,100]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [1])
]

# 45 (r35) - Test comm costs with a few heuristics
data_config[9] = [
('process_no', [45]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0, 1, 5, 10]),
('plan_iterations', [500]),
('experience', [100]),
('trials', [50]),
('alpha', [1])
]

# 46 (r36) - Experience with a few heuristics
data_config[9] = [
('process_no', [46]),
('scenario_id', [2]),
('heuristic_id', [3,4,10,11]),
('comm_branch_factor', [3]),
('comm_iterations', [100]),
('comm_cost', [0]),
('plan_iterations', [500]),
('experience', [0, 10, 25, 100, 500]),
('trials', [50]),
('alpha', [1])
]


