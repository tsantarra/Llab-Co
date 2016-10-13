import logging


def calculate_optimal_policy():
    from domains.assembly.assembly_scenario import *
    import mdp.graph_planner as dpthts
    # Initialize map
    scenario = assembly_scenario

    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    logging.debug('Beginning search.')
    while not scenario.end(state):
        # Plan
        (action, node) = dpthts.search(state, scenario, 1000, root_node=node)
        state = scenario.transition(state, action).sample()

        logging.debug('Action: ' + str(action))
        logging.debug('New state: ' + str(state) + '\n' + str(state))

        node = [node for node in node.successors[action] if node.state == state][0]

        print(action)
        print(state)


def make_teammates():
    policy = calculate_optimal_policy()

    # do stuff to sample