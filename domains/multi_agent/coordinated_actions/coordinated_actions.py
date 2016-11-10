from agents.communication.communication_strategies import get_active_node_set
from domains.multi_agent.coordinated_actions.coordinated_actions_scenario import CoordinatedActionsScenario, RandomPolicyTeammate
from agents.communication.communicating_teammate_model import CommunicatingTeammateModel
from agents.communication.communication_scenario import communicate
from agents.modeling_agent import ModelingAgent, get_max_action
from agents.models.frequentist_model import FrequentistModel
from visualization.graph import show_graph
from mdp.action import Action

from agents.communication.communication_strategies import *


def heuristic_comm(ad_hoc_agent, agent_dict, max_queries=1):
    policy_graph_root = ad_hoc_agent.policy_graph_root

    original_agent_action = get_max_action(policy_graph_root, ad_hoc_agent.identity)
    print('ORIGINAL:', original_agent_action)

    # The communication loop.
    eligible_states = set(node.state['World State'] for node in get_active_node_set(policy_graph_root))
    query_set = set()
    while eligible_states and len(query_set) < max_queries:
        # Decide on a query
        query_action = weighted_entropy(root=policy_graph_root, eligible_states=eligible_states)

        # Response bookkeeping
        target_name = query_action.agent
        target_entity = agent_dict[target_name]
        query_set.add(query_action.state)

        # recalculate eligible set, as some nodes may be unreachable now.
        eligible_states = set(node.state['World State'] for node in get_active_node_set(policy_graph_root))
        eligible_states -= query_set

        # Response & output
        response = target_entity.get_action(query_action.state)

        print('Query:',query_action.state)
        print('Response:', response)

        # Update model
        new_model = ad_hoc_agent.model_state[target_name].communicated_policy_update([(query_action.state, response)])
        ad_hoc_agent.model_state = ad_hoc_agent.model_state.update({target_name: new_model})

        # Update agent's policy
        new_root_state = policy_graph_root.state.update({'Models': ad_hoc_agent.model_state})
        ad_hoc_agent.update_policy_graph(policy_graph_root, new_root_state)

        # Check agent's policy
        new_agent_action = get_max_action(policy_graph_root, ad_hoc_agent.identity)
        print('NEW AGENT ACTION:', new_agent_action)
        print('NEW EV:', policy_graph_root.future_value)


def run_coordinated_actions():
    # Scenario
    scenario = CoordinatedActionsScenario(action_set='AB', rounds=10)
    state = scenario.initial_state()

    # Agents
    teammate = RandomPolicyTeammate(actions=scenario.action_set, rounds=scenario.rounds)
    agent = ModelingAgent(scenario, 'Agent', {'Teammate': CommunicatingTeammateModel(teammate_model=FrequentistModel(scenario, 'Teammate'), scenario=scenario)})
    agent_dict = {'Agent': agent, 'Teammate': teammate}

    first_comm_turn = True
    # Execution loop
    while not scenario.end(state):
        # Have the agents select actions
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

        print('EV:', agent.policy_graph_root.future_value)

        #agent_action = communicate(agent=agent, agent_dict=agent_dict, passes=1000, comm_cost=0)
        #action = action.update({'Agent': agent_action})

        if first_comm_turn:
            heuristic_comm(agent, agent_dict, max_queries=50)
            agent_action = get_max_action(agent.policy_graph_root, agent.identity)
            action = action.update({'Agent': agent_action})

            print('New EV:', agent.policy_graph_root.future_value)
            first_comm_turn = False

        new_state = scenario.transition(state, action).sample()

        # Update agent info
        for participating_agent in agent_dict.values():
            participating_agent.update(state, action)

        # Output
        print('-----------------')
        print('Action:', action)
        print('New State')
        print(new_state)
        print('-----------------')

        state = new_state


if __name__ == '__main__':
    run_coordinated_actions()
