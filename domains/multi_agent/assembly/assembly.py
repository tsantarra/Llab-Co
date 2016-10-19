import sys
import traceback
import logging

from domains.multi_agent.assembly.assembly_scenario import *
from ad_hoc.communication.communicating_teammate import CommunicatingTeammate
from ad_hoc.communication.communication_scenario import communicate
from ad_hoc.modeling_agent import ModelingAgent
from mdp.action import Action


def centralized_assembly():
    """
    Runs the cops and robbers scenario using a centralized planner that controls all agents.
    """
    # Local imports
    from mdp.graph_planner import search, greedy_action

    # Initialize map
    scenario = assembly_scenario
    state = scenario.initial_state()
    print('Initial state:\n', state)

    # Main loop
    util = 0
    node = None
    while not scenario.end(state):
        # Plan
        node = search(state, scenario, 1000, root_node=node)
        action = greedy_action(node)

        # Transition
        new_state = scenario.transition(state, action).sample()
        util += scenario.utility(state, action, new_state)
        node = node.find_matching_successor(new_state, action)

        # Output
        print(action)
        print(new_state)
        print('Util: ', util)

        state = new_state


def ad_hoc_assembly():
    # Initialize scenario and beginning state.
    scenario = assembly_scenario
    state = scenario.initial_state()
    logging.debug('Initial state:\n' + str(state))

    # Agents
    teammate = None
    teammate_model = CommunicatingTeammate(teammate_model= None, scenario=scenario)
    agent = ModelingAgent(scenario, 'A', {'P': teammate_model})
    agents = {'A': agent, 'P': teammate}

    # Main loop
    logging.debug('Beginning simulation.')
    while not scenario.end(state):
        # Have the agents select actions
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agents.items()})

        #  action = communicate(state, agent, agent_dict, 200)
        #  show_graph(agent.policy_graph_root)

        new_state = scenario.transition(state, action).sample()

        # Update agent info
        for participating_agent in agents.values():
            participating_agent.update(state, action)

        # Output
        print('Action:', action)
        print('New State')
        print(new_state)
        print('-----------------')


if __name__ == "__main__":
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        centralized_assembly()
        #ad_hoc_assembly()

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
