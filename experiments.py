from domains.multi_agent.assembly.assembly_scenario import AssemblyScenario

from agents.sampled_policy_teammate import SampledPolicyTeammate
from agents.communication.communicating_teammate_model import CommunicatingTeammateModel
from agents.communication.communication_scenario import communicate as full_communicate
from agents.communication.communication_strategies import *

from agents.modeling_agent import ModelingAgent
from agents.models.experts_model import ExpertsModel
from mdp.action import Action

import sys
import csv

from mdp.state import State


def initialize_agents(scenario, num_models):
    teammate = SampledPolicyTeammate('Agent2', scenario, rationality=1.0, min_graph_iterations=100)

    model_set = [SampledPolicyTeammate('Agent2', scenario, rationality=1.0, min_graph_iterations=100) for _ in range(num_models)]
    expert_model = ExpertsModel(scenario, {model: 1./num_models for model in model_set}, 'Agent2')
    comm_model = CommunicatingTeammateModel(expert_model, scenario)

    ad_hoc_agent = ModelingAgent(scenario, 'Agent1', {'Agent2': comm_model}, iterations=100000)

    return ad_hoc_agent, teammate


def run_initial_attempts(scenario, ad_hoc_agent, teammate, num_attempts):
    agent_dict = {'Agent1': ad_hoc_agent, 'Agent2': teammate}
    for run in range(num_attempts):
        # Initial state
        state = scenario.initial_state()

        while not scenario.end(state):
            action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

            new_state = scenario.transition(state, action).sample()

            # Update agent info
            for participating_agent in agent_dict.values():
                participating_agent.update(state, action)

            state = new_state

        ad_hoc_agent.policy_graph_root = None


def run_comm_attempt(scenario, ad_hoc_agent, teammate, comm_method, stop_val, max_queries=50):
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
        log_results(len(query_set), exp_util)

        # Output and termination check
        print('val:', exp_util)
        if abs(exp_util - stop_val) < 10e-6:
            break

        # recalculate eligible set, as some nodes may be unreachable now.
        query_set.add(query_action.state)
        eligible_states = set(node.state['World State'] for node in get_active_node_set(policy_graph_root))
        eligible_states -= query_set

        print('new comm states:', len(eligible_states))


def perfect_knowledge_val(scenario, ad_hoc_agent, teammate):
    agent = ad_hoc_agent.copy()
    agent.policy_graph_root = None
    agent.model_state = State({teammate.identity: ExpertsModel(scenario, {teammate: 1.0}, teammate.identity)})

    state = scenario.initial_state()
    agent.get_action(state)
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

    #csv_dir = sys.argv[1]

    num_prior_models = [ 10, 20]
    current_exp = [0, 5, 20]
    comm_methods = {weighted_variance:'weighted variance', variance_in_util:'variance', weighted_entropy: 'weighted entropy', entropy: 'entropy'}

    for num_models, num_attempts in product(num_prior_models, current_exp):

        try:
            # Initialize scenario and agents. These will be kept constant across comm methods.
            scenario = AssemblyScenario(num_components=3, rounds=4, ingredients_per_recipe=4)
            ad_hoc_agent, teammate = initialize_agents(scenario, num_models)

            # Perfect knowledge exp util
            perfect_util = perfect_knowledge_val(scenario, ad_hoc_agent, teammate)
            ad_hoc_agent.policy_graph_root = None
            print('perfect util:', perfect_util)

            # Run initial attempts, keeping the agent's model
            run_initial_attempts(scenario, ad_hoc_agent, teammate, num_attempts)

            for comm_method in comm_methods:
                # Bookkeeping
                global params
                params = [num_models, num_attempts, comm_methods[comm_method]]
                print(params)

                # Create copy for comparison, as they'll diverge with different information.
                test_agent = ad_hoc_agent.copy()

                # Run current attempt, communicating in the first round
                run_comm_attempt(scenario, test_agent, teammate, comm_method, stop_val=perfect_util, max_queries=5000)

        except KeyboardInterrupt:
            print('KEYBOARD INTERRUPT')
            raise

        except:
            print('error:', sys.exc_info()[0])
            raise
