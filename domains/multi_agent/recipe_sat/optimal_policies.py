from itertools import product
from random import sample, randint

from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario
from mdp.graph_planner import search, map_graph


def calculate_optimal_policies(scenario):
    # Initial state from scenario
    state = scenario.initial_state()
    print('Initial state:\n', state)

    # Plan from initial state. Map entire policy graph.
    root_node = search(state, scenario, heuristic=lambda scenario_state: 0)
    graph_map = map_graph(root_node)

    # For each node, identify the max policy action, then collect all actions with identical utility.
    optimal_policy_space = {}
    for state, node in graph_map.items():
        action_values = node.action_values()
        max_action, max_val = max(action_values.items(), key=lambda pair: pair[1], default=(None, 0))
        optimal_policy_space[state] = [action for action, val in action_values.items()
                                       if abs(val - max_val) < 10e-6]

        if not optimal_policy_space[state]:
            optimal_policy_space[state] = [None]

    return optimal_policy_space


def reservoir_sample(policies, sample_size):
    reservoir = []
    for _ in range(sample_size):
        reservoir.append(next(policies))

    for i, policy in enumerate(policies):
        j = randint(1, sample_size+i)
        if j <= sample_size:
            reservoir[j-1] = policy

    return reservoir


def get_team_policies(scenario, num_policies):
    policy_space = calculate_optimal_policies(scenario)

    states = policy_space.keys()
    policies = product(*policy_space.values())  # must unpack values

    return [dict(zip(states, pol)) for pol in reservoir_sample(policies, num_policies)]


if __name__ == '__main__':
    scenario = RecipeScenario(num_conditions=10, num_agents=2, num_valid_recipes=2, recipe_size=4)

    print('\n\n'.join(str(pol) for pol in get_team_policies(scenario, 4)))
