from domains.multi_agent.coordinated_actions.coordinated_actions_scenario import CoordinatedActionsScenario, RandomPolicyTeammate
from ad_hoc.communication.communicating_teammate import CommunicatingTeammate
from ad_hoc.communication.communication_scenario import communicate
from ad_hoc.modeling_agent import ModelingAgent
from ad_hoc.models.frequentist_model import FrequentistModel
from visualization.graph import show_graph
from mdp.action import Action


def run_coordinated_actions():
    # Scenario
    scenario = CoordinatedActionsScenario(action_set='AB', rounds=3)
    state = scenario.initial_state()

    # Agents
    teammate = RandomPolicyTeammate(actions=scenario.action_set, rounds=scenario.rounds)
    agent = ModelingAgent(scenario, 'Agent', {'Teammate': CommunicatingTeammate(teammate_model=FrequentistModel(scenario, 'Teammate'), scenario=scenario)})
    agent_dict = {'Agent': agent, 'Teammate': teammate}

    # Execution loop
    while not scenario.end(state):
        # Have the agents select actions
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

        agent_action = communicate(agent=agent, agent_dict=agent_dict, passes=1000, comm_cost=0)
        action = action.update({'Agent': agent_action})

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
