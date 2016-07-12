import logging
import sys
import traceback

from visualization.graph import show_graph


def carpy_dpthts():
    from domains.cops_and_robbers.cops_and_robbers_scenario import show_state, initialize_maze, \
        heuristic, cops_and_robbers_scenario
    import mdp.graph_planner as dpthts
    # Initialize map
    initialize_maze('./mazes/a.maze')
    scenario = cops_and_robbers_scenario

    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    logging.debug('Beginning search.')
    while not scenario.end(state):
        # Plan
        (action, node) = dpthts.search(state, scenario, 1000, heuristic=heuristic, root_node=node)
        state = scenario.transition(state, action).sample()
        show_graph(node, width=10, height=10)

        logging.debug('Action: ' + str(action))
        logging.debug('New state: ' + str(state) + '\n' + show_state(state))

        node = [node for node in node.successors[action] if node.state == state][0]

        print(action)
        print(show_state(state))


def multiagent_carpy():
    from domains.cops_and_robbers.cops_and_robbers_scenario import heuristic, initialize_maze, cops_and_robbers_scenario, show_state
    from domains.cops_and_robbers.teammate_models import build_experts_model, AstarTeammate
    from multiagent.modeling_agent import ModelingAgent
    from random import choice

    # Initialize scenario and beginning state.
    maze = initialize_maze('./mazes/a.maze')
    scenario = cops_and_robbers_scenario
    state = scenario.initial_state()
    logging.debug('Initial state:\n' + str(state))

    # Agents
    teammate = AstarTeammate(scenario, target=choice([key for key in state if 'Robber' in key]), maze=maze)
    agent = ModelingAgent(scenario, 'A', {'P': build_experts_model(scenario, maze, state)}, heuristic=heuristic)
    agents = {'A': agent, 'P': teammate}

    # Main loop
    logging.debug('Beginning simulation.')
    while not scenario.end(state):
        current_agent = agents[state['Turn']]

        action = current_agent.get_action(state)
        if current_agent == agent: print(agent.policy_graph_root.finite_horizon_string(horizon=4))
        new_state = scenario.transition(state, action).sample()

        logging.debug('Action: ' + str(state['Turn']) + '\t' + str(action))
        logging.debug('New state: ' + show_state(new_state) + '\n')

        for participating_agent in agents.values():
            participating_agent.update(state['Turn'], state, action, new_state)

        print(show_state(new_state))
        print(agent.models)

        state = new_state


if __name__ == "__main__":
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        multiagent_carpy()
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
