from domains.multi_agent.coordinated_actions.coordinated_actions_scenario import CoordinatedActionsScenario, SampledPolicyTeammate
from multiagent.communication.communicating_teammate import CommunicatingTeammate
from multiagent.communication.communication_scenario import communicate
from multiagent.modeling_agent import ModelingAgent
from multiagent.models.frequentist_model import FrequentistModel
from visualization.graph import show_graph
from mdp.action import Action


def run_coordinated_actions():
    # Scenario
    scenario = CoordinatedActionsScenario(action_set='AB', rounds=3)
    state = scenario.initial_state()

    # Agents
    teammate = SampledPolicyTeammate(actions=scenario.action_set, rounds=scenario.rounds)
    agent = ModelingAgent(scenario, 'Agent', {'Teammate': CommunicatingTeammate(teammate_model=FrequentistModel(scenario, 'Teammate'), scenario=scenario)})
    agent_dict = {'Agent': agent, 'Teammate': teammate}

    # Execution loop
    while not scenario.end(state):
        # Have agent act
        action = Action({agent_name: agent.get_action(state) for agent_name, agent in agent_dict.items()})

        print('Action:', action)

        #if state['Turn'] == 'Agent':
        #    action = communicate(state, agent, agent_dict, 200)
        #    show_graph(agent.policy_graph_root)

        new_state = scenario.transition(state, action).sample()

        # Output
        print('New State')
        print(new_state)
        print('-----------------')


        # Update agent info
        for participating_agent in agent_dict.values():
            participating_agent.update(state, action, new_state)

        state = new_state


if __name__ == '__main__':
    run_coordinated_actions()