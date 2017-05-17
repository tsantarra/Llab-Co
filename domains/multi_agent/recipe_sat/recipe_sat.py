from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario
from domains.profiler.profile_test import profile


def centralized_recipe_sat():
    # Local imports
    from mdp.graph_planner import search, greedy_action

    # Initialize map
    scenario = RecipeScenario(num_conditions=14, num_agents=2, num_valid_recipes=15, recipe_size=4)
    print('\n'.join(str(recipe) for recipe in scenario.recipes))
    state = scenario.initial_state()
    print('Initial state:\n', state)

    # Main loop
    util = 0
    node = None
    while not scenario.end(state):
        # Plan
        node = search(state, scenario, 1000, root_node=node, heuristic=lambda state: 0)
        action = greedy_action(node)
        print('Subgraph size: ', node.reachable_subgraph_size())

        # Transition
        new_state = scenario.transition(state, action).sample()
        util += scenario.utility(state, action, new_state)
        node = node.find_matching_successor(new_state, action)

        # Output
        print(action)
        print(new_state)
        print('Util: ', util)

        state = new_state


if __name__ == '__main__':
    centralized_recipe_sat()