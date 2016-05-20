from domains.coffee_robot.coffee_robot_scenario import coffee_robot_scenario


def coffeeRobotBFS(scenario):
    from mdp.solvers.bfs import breadth_first_search
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    # Plan
    plan = breadth_first_search(state, scenario)

    # Execute plan
    while plan:
        action = plan.pop(0)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)


def coffeeRobot_dpthts(scenario):
    import mdp.solvers.thts_dp as dpthts
    from visualization.graph import show_graph
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = dpthts.graph_search(state, scenario, 1000, root_node=node)
        state = scenario.transition(state, action).sample()

        show_graph(node, width=10, height=10)
        node = [n for n in node.successors[action] if n.state == state][0]

        print(action)
        print(state)


def coffeeRobotMCTS(scenario):
    from mdp.solvers.mcts import mcts
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = mcts(state, scenario, 1000)
        state = scenario.transition(state, action).sample()

        # print(node.TreeToString(3))
        print(action)
        print(state)


def coffeeRobotVI(scenario):
    from mdp.solvers.vi import value_iteration

    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    while not scenario.end(state):
        # Plan
        action = value_iteration(state, scenario, horizon=6)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)

if __name__ == "__main__":
    # Run scenario
    #coffeeRobotVI(coffee_robot_scenario)
    #coffeeRobotBFS(coffee_robot_scenario)
    #coffeeRobotMCTS(coffee_robot_scenario)
    coffeeRobot_dpthts(coffee_robot_scenario)
