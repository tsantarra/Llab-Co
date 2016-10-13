import logging
import sys
import traceback

from domains.assembly.assembly_scenario import *
from domains.assembly.assembly_teammates import *
from multiagent.modeling_agent import ModelingAgent
from visualization.graph import show_graph
from multiagent.communication.communication_scenario import communicate
from multiagent.communicating_teammate import CommunicatingTeammate


def multiagent_assembly():
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
        current_agent = agents[state['Turn']]
        action = current_agent.get_action(state)

        print('Turn:', state['Turn'])
        print('Action:', action)

        if state['Turn'] == 'A':
            action = communicate(state, agent, agents, 5)
            show_graph(agent.policy_graph_root)

        new_state = scenario.transition(state, action).sample()

        # Output
        logging.debug('Action: ' + str(state['Turn']) + '\t' + str(action))
        print(agent.model_state)

        for participating_agent in agents.values():
            participating_agent.update(state['Turn'], state, action, new_state)

        state = new_state


if __name__ == "__main__":
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        multiagent_assembly()
    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')
    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
