from domains.coordinated_actions.coordinated_actions_scenario import CoordinatedActionsScenario, SampledPolicyTeammate
from multiagent.modeling_agent import ModelingAgent
from multiagent.frequentist_model import FrequentistModel
from visualization.graph import show_graph
from multiagent.communication.communication_scenario import communicate
from multiagent.communicating_teammate import CommunicatingTeammate


def run_coordinated_actions():
    # Scenario
    scenario = CoordinatedActionsScenario(action_set='AB', rounds=3)
    state = scenario.initial_state()

    # Agents
    teammate = SampledPolicyTeammate(actions=scenario.action_set, rounds=scenario.rounds)
    agent = ModelingAgent(scenario, 'Agent', {'Teammate': CommunicatingTeammate(teammate_model=FrequentistModel(scenario))})
    agent_dict = {'Agent': agent, 'Teammate': teammate}

    # Execution loop
    while not scenario.end(state):
        # Have agent act
        current_agent = agent_dict[state['Turn']]
        action = current_agent.get_action(state)

        print('Turn:', state['Turn'])
        print('Action:', action)

        #if state['Turn'] == 'Agent':
        #    show_graph(agent.policy_graph_root)

        if state['Turn'] == 'Agent':
            action = communicate(state, agent, agent_dict, 20)

        if state['Turn'] == 'Agent':
            show_graph(agent.policy_graph_root)

        new_state = scenario.transition(state, action).sample()

        # Output
        print('Action:', action)
        print('New State')
        print(new_state)
        print('-----------------')


        # Update agent info
        for participating_agent in agent_dict.values():
            participating_agent.update(state['Turn'], state, action, new_state)

        state = new_state


if __name__ == '__main__':
    run_coordinated_actions()