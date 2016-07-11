from domains.grid.grid_scenario import grid_scenario
from mdp.thts_dp import graph_search


def grid_test(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    while not scenario.end(state):
        # Plan
        (action, node) = graph_search(state, scenario, 10000)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)
        #print(node.tree_to_string(horizon=3))


if __name__ == "__main__":
    grid_test(grid_scenario)
