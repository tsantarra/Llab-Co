import logging
import sys
import traceback

from domains.multi_agent.cops_and_robbers.cops_and_robbers_scenario import CopsAndRobbersScenario
from mdp.action import Action


def centralized_carpy():
    """
    Runs the cops and robbers scenario using a centralized planner that controls all agents.
    """
    # Local imports
    from mdp.graph_planner import search, greedy_action

    # Initialize map
    scenario = CopsAndRobbersScenario('a.maze', last_round=10)
    state = scenario.initial_state()
    print('Initial state:\n', state)

    # Main loop
    node = None
    while not scenario.end(state):
        # Plan
        node = search(state, scenario, 1000, heuristic=scenario.heuristic, root_node=node)
        action = greedy_action(node)

        # Transition
        state = scenario.transition(state, action).sample()
        node = node.find_matching_successor(state, action)

        # Output
        print(action)
        print(scenario.show_state(state))


def ad_hoc_carpy():
    """
    Runs the cops and robbers scenario using a modeling agent and an 'unknown' agent.
    """
    # Local imports
    from domains.multi_agent.cops_and_robbers.teammate_models import build_experts_model, AstarTeammate
    from agents.modeling_agent import ModelingAgent
    from agents.communication.communicating_teammate_model import CommunicatingTeammateModel

    from random import choice

    # Initialize map
    scenario = CopsAndRobbersScenario('a.maze')
    state = scenario.initial_state()
    logging.debug('Initial state:\n' + str(state))

    # Agents
    teammate = AstarTeammate(scenario=scenario,
                             target=choice([key for key in state if 'Robber' in key]),
                             maze=scenario.maze)
    teammate_model = CommunicatingTeammateModel(teammate_model=build_experts_model(scenario, scenario.maze, state),
                                                scenario=scenario)

    agent = ModelingAgent(scenario=scenario,
                          identity='A',
                          models={'S': teammate_model})
    agents = {'A': agent, 'S': teammate}

    # Main loop
    logging.debug('Beginning simulation.')
    while not scenario.end(state):
        # Plan
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agents.items()})

        # Communicate
        #  action = communicate(state, agent, agents, 200)
        #  show_graph(agent.policy_graph_root)

        # Transition
        new_state = scenario.transition(state, action).sample()

        # Update
        for participating_agent in agents.values():
            participating_agent.update(state, action)

        # Output
        logging.debug('Action: ' + str(action))
        logging.debug('New state: ' + scenario.show_state(new_state) + '\n')
        print('Action:', action)
        print(scenario.show_state(new_state))
        print(agent.model_state)

        state = new_state


if __name__ == "__main__":
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        centralized_carpy()
        #ad_hoc_carpy()

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
