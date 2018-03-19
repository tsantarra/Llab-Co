from domains.multi_agent.assembly.assembly_scenario import AssemblyScenario
from agents.sampled_policy_teammate import SampledPolicyTeammate
from agents.communication.communicating_teammate_model import CommunicatingTeammateModel
from agents.communication.graph_utilities import get_active_node_set
from agents.modeling_agent import ModelingAgent, get_max_action
from agents.models.experts_model import ExpertsModel
from agents.models.chinese_restaurant_process_model import ChineseRestaurantProcessModel
from mdp.action import Action
from mdp.state import State

import sys
import csv


def initialize_agents(scenario, num_models):  # FOR ONE SHOT TESTING!
    """
    Sample num_models worth of teammate policies (probably a better way to do this, caching the graph TODO).
    Add them to Chinese Restaurant Process to generate a prior.
    Use a communicating experts model as the base model for the ad hoc agent.
    """
    chinese_restaurant_process = ChineseRestaurantProcessModel('Agent2', scenario, 1)
    for _ in range(num_models):
        chinese_restaurant_process.add_observed_policy(SampledPolicyTeammate('Agent2', scenario, rationality=1.0))

    comm_model = CommunicatingTeammateModel(chinese_restaurant_process.get_expert_prior(), scenario)

    return ModelingAgent(scenario, 'Agent1', {'Agent2': comm_model}, iterations=10000), \
           SampledPolicyTeammate('Agent2', scenario, rationality=1.0, min_graph_iterations=100)


def run_comm_attempt(scenario, ad_hoc_agent, teammate, comm_method, max_queries=50):
    agent_dict = {'Agent1': ad_hoc_agent, 'Agent2': teammate}
    state = scenario.initial_state()

    action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})
    policy_graph_root = ad_hoc_agent.policy_graph_root

    original_agent_action = get_max_action(policy_graph_root, ad_hoc_agent.identity)

    # The communication loop.
    eligible_states = set(node.state['World State'] for node in get_active_node_set(policy_graph_root))
    query_set = set()
    print('comm states:', len(eligible_states))
    while eligible_states and len(query_set) < max_queries:
        # Decide on a query
        query_action = comm_method(root=policy_graph_root, eligible_states=eligible_states)

        # Response bookkeeping
        target_name = query_action.agent
        target_entity = agent_dict[target_name]

        # Response & output
        response = target_entity.get_action(query_action.state)

        # Update model
        new_model = ad_hoc_agent.model_state[target_name].communicated_policy_update([(query_action.state, response)])
        ad_hoc_agent.model_state = ad_hoc_agent.model_state.update({target_name: new_model})

        # Update agent's policy
        new_root_state = policy_graph_root.state.update({'Models': ad_hoc_agent.model_state})
        ad_hoc_agent.update_policy_graph(policy_graph_root, new_root_state)

        # Log
        exp_util = ad_hoc_agent.policy_graph_root.future_value
        # log_results(len(query_set), exp_util)

        # Output
        print('val:', exp_util)

        # recalculate eligible set, as some nodes may be unreachable now.
        query_set.add(query_action.state)
        eligible_states = set(node.state['World State'] for node in get_active_node_set(policy_graph_root))
        eligible_states -= query_set

        print('new comm states:', len(eligible_states))


def perfect_knowledge_val(scenario, ad_hoc_agent, teammate):
    agent = ad_hoc_agent.copy()
    agent.policy_graph_root = None
    agent.model_state = State({teammate.identity: ExpertsModel(scenario, {teammate: 1.0}, teammate.identity)})
    agent.get_action(scenario.initial_state())

    return agent.policy_graph_root.future_value


def log_results(num_comm, exp_util):
    dir = "D://Google Drive//aamas_results//"
    global params

    with open(dir + 'results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(params + [num_comm, exp_util, ])


if __name__ == '__main__':
    from functools import partial
    from itertools import product

    num_prior_models = range(0, 100, 5)
    comm_methods = {weighted_variance:'weighted variance', variance_in_util:'variance', weighted_entropy: 'weighted entropy', entropy: 'entropy'}

    for num_models in num_prior_models:

        try:
            # Initialize scenario and agents. These will be kept constant across comm methods.
            scenario = AssemblyScenario(num_components=3, rounds=4, ingredients_per_recipe=4)
            ad_hoc_agent, teammate = initialize_agents(scenario, num_models)
            ad_hoc_agent.policy_graph_root = None

            # Perfect knowledge exp util
            perfect_util = perfect_knowledge_val(scenario, ad_hoc_agent, teammate)
            print('perfect util:', perfect_util)

            for comm_method in comm_methods:
                # Bookkeeping
                global params
                params = [num_models, comm_methods[comm_method]]
                print(params)

                # Create copy for comparison, as they'll diverge with different information.
                test_agent = ad_hoc_agent.copy()

                # Run current attempt, communicating in the first round
                run_comm_attempt(scenario, test_agent, teammate, comm_method, max_queries=5000)

        except KeyboardInterrupt:
            print('KEYBOARD INTERRUPT')
            raise

        except:
            print('error:', sys.exc_info()[0])
            raise
