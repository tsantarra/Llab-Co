import logging
import sys
import traceback

from agents.communication.communicating_teammate_model import CommunicatingTeammateModel
from agents.modeling_agent import ModelingAgent
from domains.multi_agent.assembly.assembly_scenario import *
from mdp.action import Action


def get_node_set(root):
    """
    Traverse graph. Return set of nodes.
    """
    process_list = [root]
    added = {root}

    # Queue all nodes in tree accordingto depth
    while process_list:
        node = process_list.pop()

        for successor in node.successor_set():
            if successor not in added:
                # Add to process list and added set
                process_list.append(successor)
                added.add(successor)

    return added


def check_graph_sizes():
    # Local imports
    from mdp.graph_planner import search
    from itertools import product
    import time

    components = [3]
    recipes = [3,5]
    rounds = [7,8,9,11]
    ingredients = [5,10]

    for comp, rec, rnds, ingred in sorted(product(components, recipes, rounds, ingredients), key=lambda s: max(s)):
        # Initialize map
        scenario = AssemblyScenario(comp, rec, rnds, ingred)
        state = scenario.initial_state()

        print('components:', comp, 'recipes:', rec,'rounds:', rnds, 'ingredients:', ingred)
        start = time.time()
        node = search(state, scenario, 100000, root_node=None, heuristic=lambda state: 0)
        diff = time.time() - start
        nodes = get_node_set(node)

        print('Size:', len(nodes))
        print('Val:', node.future_value)
        print('Time:', diff)


def centralized_assembly():
    # Local imports
    from mdp.graph_planner import search, greedy_action

    # Initialize map
    scenario = AssemblyScenario()
    state = scenario.initial_state()
    print('Initial state:\n', state)

    # Main loop
    util = 0
    node = None
    while not scenario.end(state):
        # Plan
        node = search(state, scenario, 1000, root_node=node, heuristic=lambda state: 0)
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


def sampled_assembly():
    """
    """
    from agents.sampled_policy_teammate import SampledPolicyTeammate
    scenario = AssemblyScenario()
    agent_dict = {'Agent1': SampledPolicyTeammate('Agent1', scenario, 10, 1000),
                  'Agent2': SampledPolicyTeammate('Agent2', scenario, 10, 1000)}

    state = scenario.initial_state()

    print('Initial state:\n', state)

    # Main loop
    util = 0
    while not scenario.end(state):
        # Plan
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

        # Transition
        new_state = scenario.transition(state, action).sample()
        util += scenario.utility(state, action, new_state)

        # Output
        print(action)
        print(new_state)
        print('Util: ', util)

        state = new_state


def ad_hoc_assembly():
    # Initialize scenario and beginning state.
    scenario = AssemblyScenario()
    state = scenario.initial_state()
    logging.debug('Initial state:\n' + str(state))

    # Agents
    teammate = None
    teammate_model = CommunicatingTeammateModel(teammate_model= None, scenario=scenario)
    agent = ModelingAgent(scenario, 'A', {'P': teammate_model}, heuristic=lambda assembly_state: 0)
    agent_dict = {'A': agent, 'P': teammate}

    # Main loop
    logging.debug('Beginning simulation.')
    while not scenario.end(state):
        # Have the agents select actions
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

        #  action = communicate(state, agent, agent_dict, 200)
        #  show_graph(agent.policy_graph_root)

        new_state = scenario.transition(state, action).sample()

        # Update agent info
        for participating_agent in agent_dict.values():
            participating_agent.update(state, action)

        # Output
        print('Action:', action)
        print('New State')
        print(new_state)
        print('-----------------')


if __name__ == "__main__":
    logging.basicConfig(filename=__file__[:-3] + '.log', filemode='w', level=logging.DEBUG)

    try:
        #check_graph_sizes()
        sampled_assembly()
        #centralized_assembly()
        #ad_hoc_assembly()

    except KeyboardInterrupt:
        print('ctrl-c, leaving ...')

    except Exception:
        traceback.print_exc(file=sys.stdout)
        logging.exception("Error.")
