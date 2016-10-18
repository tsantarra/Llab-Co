import logging
import sys
import traceback

from visualization.graph import show_graph


def carpy_dpthts():
    from domains.multi_agent.cops_and_robbers.cops_and_robbers_scenario import show_state, initialize_maze, \
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
    from domains.multi_agent.cops_and_robbers.cops_and_robbers_scenario import heuristic, initialize_maze, cops_and_robbers_scenario, show_state
    from domains.multi_agent.cops_and_robbers.teammate_models import build_experts_model, AstarTeammate
    from multiagent.modeling_agent import ModelingAgent
    from random import choice
    from visualization.graph import show_graph
    from multiagent.communication.communication_scenario import communicate
    from multiagent.communication.communicating_teammate import CommunicatingTeammate

    # Initialize scenario and beginning state.
    maze = initialize_maze('./mazes/a.maze')
    scenario = cops_and_robbers_scenario
    state = scenario.initial_state()
    logging.debug('Initial state:\n' + str(state))

    # Agents
    teammate = AstarTeammate(scenario, target=choice([key for key in state if 'Robber' in key]), maze=maze)
    teammate_model = CommunicatingTeammate(teammate_model=build_experts_model(scenario, maze, state), scenario=scenario)
    agent = ModelingAgent(scenario, 'A', {'P': teammate_model}, heuristic=heuristic)
    agents = {'A': agent, 'P': teammate}

    # Main loop
    logging.debug('Beginning simulation.')
    while not scenario.end(state):
        current_agent = agents[state['Turn']]
        action = current_agent.get_action(state)

        print('Turn:', state['Turn'])
        print('Action:', action)

        if state['Turn'] == 'A':
            action = communicate(state, agent, agents, 200)
            show_graph(agent.policy_graph_root)

        new_state = scenario.transition(state, action).sample()

        # Output
        logging.debug('Action: ' + str(state['Turn']) + '\t' + str(action))
        logging.debug('New state: ' + show_state(new_state) + '\n')
        print(show_state(new_state))
        print(agent.model_state)

        for participating_agent in agents.values():
            participating_agent.update(state['Turn'], state, action, new_state)

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
