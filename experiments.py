from agents.communication.communication_scenario import communicate
from agents.communication.communication_strategies import *
from agents.modeling_agent import ModelingAgent
from agents.models.chinese_restaurant_process_model import SparseChineseRestaurantProcessModel
from agents.sampled_policy_teammate import SampledTeammateGenerator
from agents.communication.communicating_teammate_model import CommunicatingTeammateModel
from agents.models.policy_distribution import PolicyDistributionModel
from mdp.state import State
from mdp.action import Action

from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario
from domains.multi_agent.cops_and_robbers.cops_and_robbers_scenario import CopsAndRobbersScenario

from collections import namedtuple
from utils.log_config import setup_logger
from math import inf
import sys
import logging
import json
import pickle
import os.path
import gzip

logger = logging.getLogger()

Parameters = namedtuple('Parameters', [
                                       'process_no',            # 0
                                       'scenario_id',           # 1
                                       'heuristic_id',          # 2
                                       'comm_branch_factor',    # 3
                                       'comm_iterations',       # 4
                                       'comm_cost',             # 5
                                       'plan_iterations',       # 6
                                       'experience',            # 7
                                       'trials',                # 8
                                       'alpha',                 # 9
                                       'policy_cap',            # 10
                                       ])

scenarios = [RecipeScenario(num_conditions=7, num_agents=2, num_valid_recipes=1, recipe_size=5),    # 0
             CopsAndRobbersScenario(filename='a.maze', last_round=13),                              # 1
             CopsAndRobbersScenario(filename='small.maze', last_round=5),                           # 2
             CopsAndRobbersScenario(filename='sidekick_first.maze', last_round=10),                 # 3
             CopsAndRobbersScenario(filename='agent_first.maze', last_round=10),                    # 4
             CopsAndRobbersScenario(filename='simultaneous.maze', last_round=10),                   # 5
             CopsAndRobbersScenario(filename='small2.maze', last_round=5),                          # 6
             CopsAndRobbersScenario(filename='small_sidekick_first.maze', last_round=8),            # 7
             ]

heuristics = [
              local_action_information_entropy,             # 0
              local_absolute_error,                         # 1
              local_mean_squared_error,                     # 2
              local_delta_policy_entropy,                   # 3
              local_value_of_information,                   # 4

              weighted(local_action_information_entropy),   # 5
              weighted(local_absolute_error),               # 6
              weighted(local_mean_squared_error),           # 7
              weighted(local_delta_policy_entropy),         # 8
              weighted(local_value_of_information),         # 9

              immediate_delta_policy_entropy,               # 10
              immediate_approx_value_of_information,        # 11

              random_evaluation,                            # 12
              state_likelihood,                             # 13
              weighted(random_evaluation),                  # 14
              ]


def run(parameters):
    # id -> param conversions
    scenario = scenarios[parameters.scenario_id]
    comm_heuristic = heuristics[parameters.heuristic_id]

    # Setup teammate generator
    agent_identity, teammate_identity = scenario.agents()
    teammate_generator = SampledTeammateGenerator(scenario=scenario,
                                                  identity=teammate_identity,
                                                  max_unique_policies=parameters.policy_cap
                                                                      if parameters.policy_cap != 0
                                                                      else inf)

    import gc

    for trial in range(parameters.trials):
        teammate_generator.reset_policy_set()
        chinese_restaurant_process = initialize_crp(scenario, teammate_identity, parameters.experience,
                                                    teammate_generator)
        teammate_model = PolicyDistributionModel(scenario,
                                                 teammate_identity,
                                                 chinese_restaurant_process.prior(alpha=parameters.alpha),
                                                 chinese_restaurant_process)
        teammate_model = CommunicatingTeammateModel(teammate_model, scenario)
        initial_models = {teammate_identity: teammate_model}

        ad_hoc_agent = ModelingAgent(scenario, agent_identity, initial_models, parameters.plan_iterations)
        partner = teammate_generator.sample_teammate()

        logger.info('Begin Trial!', extra={'Trial': trial})
        reward = run_experiment(scenario, ad_hoc_agent, partner, parameters.comm_cost, parameters.comm_branch_factor,
                                parameters.comm_iterations, comm_heuristic, trial)
        logger.info('End Trial', extra={'Trial': trial, 'Reward': reward})

        gc.collect()


def initialize_teammate_generator(scenario, teammate_identity, policy_cap):
    """
    Sample num_models worth of teammate policies.
    Due to the modeling needs of the scenario, the teammate model is represented as such:
        Communicating Teammate Model (stores and holds to previous policy commitments)
            Teammate Distribution Model - a Chinese Restaurant Process/distribution over
                - Multiple SampledPolicyTeammate models (one partial policy each; also used for the actual teammate)
                - One UniformPolicyTeammate model
    """
    precomputed_policy_graph_file = f'{type(scenario).__name__}.pickle'
    if os.path.isfile(precomputed_policy_graph_file):
        with gzip.open(precomputed_policy_graph_file, 'rb') as policy_file:
            teammate_generator = pickle.load(policy_file)
    else:
        teammate_generator = SampledTeammateGenerator(scenario, teammate_identity, max_unique_policies=policy_cap)
        with gzip.open(precomputed_policy_graph_file, 'wb') as policy_file:
            pickle.dump(teammate_generator, policy_file, protocol=pickle.HIGHEST_PROTOCOL)

    return teammate_generator


def initialize_crp(scenario, teammate_identity, num_initial_models, teammate_generator):
    chinese_restaurant_process = SparseChineseRestaurantProcessModel(teammate_identity, scenario,
                                                                     teammate_generator.policy_size)
    for _ in range(num_initial_models):
        chinese_restaurant_process.add_teammate_model(teammate_generator.sample_partial_policy())

    return chinese_restaurant_process


def perfect_knowledge_val(scenario, agent, teammate):
    """
    Attempts to plan given the true policy of the teammate. Returns the expected value of the policy.
    """
    agent = agent.copy()
    agent.policy_graph_root = None
    agent.model_state = State({teammate.identity: teammate})
    agent.get_action(scenario.initial_state())

    return agent.policy_graph_root.future_value


def run_experiment(scenario, agent, teammate, comm_cost, comm_branch_factor, comm_iterations, comm_heuristic, trial):
    """
    April 3, 2018
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
    state = scenario.initial_state()
    agent_name, teammate_name = scenario.agents()
    agent_dict = {agent_name: agent, teammate_name: teammate}
    utility = 0

    if hasattr(scenario, 'show_state'):
        print('Initial state:\n' + scenario.show_state(state),'\n')
    else:
        print('Initial state:\n', state, '\n')

    # Main loop
    while not scenario.end(state):
        # Have the agents select actions
        joint_action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

        # Communicate
        action, cost = communicate(scenario, agent, agent_dict,
                                      comm_planning_iterations=comm_iterations,
                                      comm_heuristic=comm_heuristic,
                                      branching_factor=comm_branch_factor,
                                      comm_cost=comm_cost,
                                      trial=trial)
        utility -= cost
        print('Utility spent in communication: ' + str(cost))
        new_joint_action = joint_action.update({agent_name: action})

        logger.info('State-Action', extra={'Trial': trial,
                                           'State': scenario._serialize_state(state),
                                           'Action': json.dumps(list(new_joint_action.items())),
                                           'EV': agent.policy_graph_root.future_value})

        # Observe
        for participating_agent in agent_dict.values():
            participating_agent.update(state, new_joint_action)

        # Update world
        new_state = scenario.transition(state, new_joint_action).sample()

        # Collect reward
        utility += scenario.utility(state, new_joint_action, new_state)

        # Output
        print('\n-----------------')
        print('Original:\t', joint_action)
        print('Updated: \t', new_joint_action)
        print('EV:\t', agent.policy_graph_root.future_value)
        print('Policy distribution length:', len(agent.model_state[teammate_name].model.policy_distribution))
        print('New Action:', new_joint_action)
        if hasattr(scenario, 'show_state'):
            print('New State:\n' + scenario.show_state(new_state))
        else:
            print('New State:\n', new_state)
        print('Utility:', utility)
        print('-----------------\n')

        # Update state
        state = new_state

    print('Final Utility:', utility, '\n')
    return utility


if __name__ == '__main__':
    assert len(sys.argv) == len(Parameters._fields) + 3, \
        'Improper arguments given: ' + ' '.join(str(i) for i in sys.argv) + '\nExpected: ' + \
        ' '.join(Parameters._fields) + ' $(Cluster) $(Process)'

    parameters = [int(arg) for arg in sys.argv[1:-2]]
    parameters = Parameters(*parameters)
    print('Parameters: ' + str(parameters))

    # Set up logger with process info.
    setup_logger(id='-'.join(sys.argv[1:]))
    logger.info('Parameters', extra=parameters._asdict())
    logger.info('Environment', extra={'Version': str(sys.version_info)})

    try:
        run(parameters)

    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT')
        raise

    except Exception as exception:
        logger.error(exception, exc_info=True)
        raise
