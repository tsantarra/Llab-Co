from domains.single_agent.grid.grid_scenario import grid_scenario
from mdp.graph_planner import search, greedy_action


def grid_test(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    while not scenario.end(state):
        # Plan
        node = search(state, scenario, 10000)
        action = greedy_action(node)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)
        #print(node.tree_to_string(horizon=3))


if __name__ == "__main__":
    grid_test(grid_scenario)
