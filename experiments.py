from agents.modeling_agent import ModelingAgent
from agents.models.chinese_restaurant_process_model import ChineseRestaurantProcessModel
from agents.sampled_policy_teammate import SampledTeammateGenerator
from agents.communication.communicating_teammate_model import CommunicatingTeammateModel
from agents.models.policy_distribution import TeammateDistributionModel

from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario

from mdp.graph_utilities import get_active_node_set
from mdp.state import State
from mdp.action import Action

import sys
import csv


def initialize_agents(scenario, num_initial_models):  # FOR ONE SHOT TESTING!
    """
    Sample num_models worth of teammate policies (probably a better way to do this, caching the graph TODO).
    Due to the modeling needs of the scenario, the teammate model is represented as such:
        Communicating Teammate Model (stores and holds to previous policy commitments)
            Teammate Distribution Model - a distribution over
                - Multiple OfflineSampledPolicyTeammate models (one policy each; also used for the actual teammate)
                - One UniformPolicyTeammate model
    """
    teammate_generator = SampledTeammateGenerator(scenario, 'Agent2')
    chinese_restaurant_process = ChineseRestaurantProcessModel('Agent2', scenario, 1)
    for _ in range(num_initial_models):
        chinese_restaurant_process.add_teammate_model(teammate_generator.sample_teammate())

    teammate_model = TeammateDistributionModel(scenario, 'Agent2', chinese_restaurant_process.prior())
    comm_model = CommunicatingTeammateModel(teammate_model, scenario)

    return ModelingAgent(scenario, 'Agent1', {'Agent2': comm_model}, iterations=10000), teammate_generator


def run_comm_attempt(agent, teammate, comm_method, max_queries=50):
    agent_dict = {'Agent1': agent, 'Agent2': teammate}
    policy_graph_root = agent.policy_graph_root

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
        new_model = agent.model_state[target_name].communicated_policy_update([(query_action.state, response)])
        agent.model_state = agent.model_state.update({target_name: new_model})

        # Update agent's policy
        new_root_state = policy_graph_root.state.update({'Models': agent.model_state})
        agent.update_policy_graph(policy_graph_root, new_root_state)

        # Log
        exp_util = agent.policy_graph_root.future_value
        # log_results(len(query_set), exp_util)

        # Output
        print('val:', exp_util)

        # recalculate eligible set, as some nodes may be unreachable now.
        query_set.add(query_action.state)
        eligible_states = set(node.state['World State'] for node in get_active_node_set(policy_graph_root))
        eligible_states -= query_set

        print('new comm states:', len(eligible_states))


def perfect_knowledge_val(scenario, agent, teammate):
    """
    Attempts to plan given the true policy of the teammate. Returns the expected value of the policy.
    """
    agent = agent.copy()
    agent.policy_graph_root = None
    agent.model_state = State({teammate.identity: teammate})
    agent.get_action(scenario.initial_state())

    return agent.policy_graph_root.future_value


def run_experiment(scenario, identity, runs):
    """
    April 3, 2018
    # Create sample teammate template (one giant graph to sample teammates from)
    population = SampledTeammateGenerator(scenario, 'Agent2')

    # For number of full runs:
    for run in range(runs):
        for comm_method in comm_methods:
            population_model = ChineseRestaurantProcessModel(identity, scenario)

            for milestone in milestones:
                # Add experience to milestone
                for _ in range(milestone):
                    population_model.add_teammate_model(population.sample_teammate())

                # Test on communicating expert distribution
                run_trial(comm_method)
    """


def run_no_comm(scenario, agent, teammate):
    state = scenario.initial_state()
    agent_dict = {'Agent1': agent, 'Agent2': teammate}

    # Main loop
    while not scenario.end(state):
        # Have the agents select actions
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

        # Communicate
        # action = communicate(state, agent, agent_dict, 200)

        # Observe
        for participating_agent in agent_dict.values():
            participating_agent.update(state, action)

        # Update world
        state = scenario.transition(state, action).sample()

        # Output
        print('\t'.join('{:4.4f}'.format(i) for i in agent.model_state['Agent2'].model.teammate_distribution.values()
                        if i > 10e-5))
        """
        print('Action:', action)
        print('New State')
        print(new_state)
        print('-----------------')
        """


if __name__ == '__main__':
    global params
    """
    comm_methods = {weighted_variance:  'weighted variance',
                    variance_in_util:    'variance',
                    weighted_entropy:   'weighted entropy',
                    entropy:            'entropy'}
    """

    for num_models in [1500]:

        try:
            # Initialize scenario and agents. These will be kept constant across comm methods.
            recipe_scenario = RecipeScenario(num_conditions=5, num_agents=2, num_valid_recipes=1, recipe_size=5)
            ad_hoc_agent, generator = initialize_agents(recipe_scenario, num_models)

            ad_hoc_agent.policy_graph_root = None
            partner = generator.sample_teammate()

            # Perfect knowledge exp util
            #perfect_util = perfect_knowledge_val(recipe_scenario, ad_hoc_agent, partner)
            #print('perfect util:', perfect_util)

            # mini test
            run_no_comm(recipe_scenario, ad_hoc_agent, partner)
            break

            # actual area to do testing
            for comm_strategy in comm_methods:
                # Bookkeeping
                params = [num_models, comm_methods[comm_strategy]]
                print(params)

                # Create copy for comparison, as they'll diverge with different information.

                test_agent = ad_hoc_agent.copy()

                # Run current attempt, communicating in the first round
                run_comm_attempt(test_agent, partner, comm_strategy, max_queries=5000)

        except KeyboardInterrupt:
            print('KEYBOARD INTERRUPT')
            raise

        except:
            print('error:', sys.exc_info()[0])
            raise
