
from Domains.Cops_and_Robbers.CopsAndRobbersScenario import show_state, cops_and_robbers_scenario, initialize_maze, heuristic

import logging, sys, traceback


def carpy_dpthts(scenario):
    import MDP.solvers.thts_dp as dpthts
    # Initialize map
    initialize_maze('C://Users//Trevor//Documents//GitHub//Llab-Co//Domains//Cops_and_Robbers//Mazes//a.maze')

    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    logging.debug('Beginning search.')
    count = 1
    while not scenario.end(state):
        # Plan
        (action, node) = dpthts.graph_search(state, scenario, 10000, heuristic=heuristic, root_node=node)
        state = scenario.transition(state, action).sample()
        node = [node for node in node.successors[action] if node.state == state][0]
        logging.debug('Action: ' + str(action))
        logging.debug('New state: ' + str(state))

        print(action)
        print(show_state(state))
        count += 1
        if count == 4:
            break
        #print(node.tree_to_string(horizon=3))


if __name__ == "__main__":
    logging.basicConfig(filename=__file__[:-3] +'.log', filemode='w', level=logging.DEBUG)
    try:
        carpy_dpthts(cops_and_robbers_scenario)
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")