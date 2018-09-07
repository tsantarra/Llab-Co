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
from log_config import setup_logger
import sys
import logging
import json

logger = logging.getLogger()

Parameters = namedtuple('Parameters', ['scenario_id',
                                       'heuristic_id',
                                       'comm_branch_factor',
                                       'comm_iterations',
                                       'comm_cost',
                                       'plan_iterations',
                                       'experience',
                                       'trials',
                                       'process_no',
                                       'osg_cluster',
                                       'osg_process'])

scenarios = [RecipeScenario(num_conditions=7, num_agents=2, num_valid_recipes=1, recipe_size=5),
             CopsAndRobbersScenario(filename='a.maze')]

heuristics = [local_information_entropy,
              local_value_of_information,
              local_absolute_error,
              local_utility_variance,
              random_evaluation,
              most_likely_next_state]  # TODO these do not have the same signature


def run():
    assert len(sys.argv) == len(Parameters._fields) + 1, 'Improper arguments given: ' + ' '.join(Parameters._fields)
    parameters = [int(arg) for arg in sys.argv[1:-2]] + list(sys.argv[-2:])
    parameters = Parameters(*parameters)
    print('Parameters: ' + str(parameters))

    # Set up logger with process info.
    setup_logger(id='-'.join(sys.argv[1:]))
    logger.info('Parameters', extra=parameters._asdict())

    # id -> param conversions
    scenario = scenarios[parameters.scenario_id]
    comm_heuristic = heuristics[parameters.heuristic_id]

    # Setup teammate generator
    agent_identity, teammate_identity = scenario.agents()
    chinese_restaurant_process, generator = initialize_agents(scenario, parameters.experience, teammate_identity)
    teammate_model = PolicyDistributionModel(scenario, teammate_identity, chinese_restaurant_process.prior(),
                                             chinese_restaurant_process)
    teammate_model = CommunicatingTeammateModel(teammate_model, scenario)
    initial_models = {teammate_identity: teammate_model}

    for trial in range(parameters.trials):
        ad_hoc_agent = ModelingAgent(scenario, agent_identity, initial_models, parameters.plan_iterations)
        partner = generator.sample_teammate()

        logger.info('Begin Trial!', extra={'Trial': trial})
        reward = run_experiment(scenario, ad_hoc_agent, partner, parameters.comm_cost, parameters.comm_branch_factor,
                                parameters.comm_iterations, comm_heuristic, trial)
        logger.info('End Trial', extra={'Trial': trial, 'Reward': reward})


def initialize_agents(scenario, num_initial_models, teammate_identity):
    """
    Sample num_models worth of teammate policies.
    Due to the modeling needs of the scenario, the teammate model is represented as such:
        Communicating Teammate Model (stores and holds to previous policy commitments)
            Teammate Distribution Model - a Chinese Restaurant Process/distribution over
                - Multiple SampledPolicyTeammate models (one partial policy each; also used for the actual teammate)
                - One UniformPolicyTeammate model
    """
    teammate_generator = SampledTeammateGenerator(scenario, teammate_identity)
    chinese_restaurant_process = SparseChineseRestaurantProcessModel(teammate_identity, scenario)
    for _ in range(num_initial_models):
        chinese_restaurant_process.add_teammate_model(teammate_generator.sample_partial_policy())

    return chinese_restaurant_process, teammate_generator


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

    # Main loop
    while not scenario.end(state):
        # Have the agents select actions
        joint_action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})
        print('Joint Action:', joint_action)

        # Communicate
        action, _, cost = communicate(scenario, agent, agent_dict,
                                comm_planning_iterations=comm_iterations,
                                comm_heuristic=comm_heuristic,
                                branching_factor=comm_branch_factor,
                                comm_cost=comm_cost,
                                trial=trial)
        utility -= cost
        print('Utility spent in communication: ' + str(cost))
        new_joint_action = joint_action.update({agent_name: action})

        logger.info('State-Action', extra={'Trial': trial, 'State': scenario._serialize_state(state),
                                           'Action': json.dumps(list(new_joint_action.items()))})

        # Observe
        for participating_agent in agent_dict.values():
            participating_agent.update(state, new_joint_action)

        # Update world
        new_state = scenario.transition(state, new_joint_action).sample()

        # Collect reward
        utility += scenario.utility(state, new_joint_action, new_state)

        # Output
        print('Policy distribution length:', len(agent.model_state[teammate_name].model.policy_distribution))
        print('New Action:', new_joint_action)
        print('New State')
        print(new_state)
        print('Utility:', utility)
        print('-----------------')

        state = new_state

    return utility


if __name__ == '__main__':
    try:
        run()

    except KeyboardInterrupt:
        print('KEYBOARD INTERRUPT')
        raise

    except Exception as exception:
        logger.error(exception, exc_info=True)
        raise
