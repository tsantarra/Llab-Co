from domains.grid.grid_scenario import grid_scenario


def grid_test_bfs(scenario):
    from mdp.solvers.bfs import breadth_first_search

    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    # Plan
    plan = breadth_first_search(state, scenario)

    # Execute plan
    while plan:
        action = plan.pop(0)
        state = scenario.transition(state, action).sample()
        print(action)
        print(state)


def grid_test_mcts(scenario):
    from mdp.solvers.mcts import mcts

    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = mcts(state, scenario, 10000)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)


def grid_test_vi(scenario):
    from mdp.solvers.vi import value_iteration

    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    while not scenario.end(state):
        # Plan
        action = value_iteration(state, scenario, horizon=51)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)


def grid_test_thts(scenario):
    from mdp.solvers.thts import tree_search

    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = tree_search(state, scenario, 10000)  # 4 ** 7 - 1
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)
        print("Tree size:", len(node))
        #print(node.tree_to_string(horizon=3))


def grid_test_dpthts(scenario):
    import mdp.solvers.thts_dp as dpthts
    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = dpthts.graph_search(state, scenario, 10000)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)
        #print(node.tree_to_string(horizon=3))


if __name__ == "__main__":
    # Run tests
    """
    print('VI Run:')
    grid_test_vi(grid_scenario)

    print('\nBFS Run:')
    grid_test_bfs(grid_scenario)

    print('\nMCTS Run:')
    grid_test_mcts(grid_scenario)

    print('\nTHTS Run:')
    grid_test_thts(grid_scenario)
    """

    print('\nDPTHTS Run:')
    grid_test_dpthts(grid_scenario)
