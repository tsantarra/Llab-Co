from domains.coordinated_actions.coordinated_actions_scenario import CoordinatedActionsScenario, SampledPolicyTeammate
from multiagent.modeling_agent import ModelingAgent
from multiagent.frequentist_model import FrequentistModel


# Scenario
scenario = CoordinatedActionsScenario(action_set='AB', rounds=5)
state = scenario.initial_state()

# Agents
teammate = SampledPolicyTeammate(actions=scenario.action_set, rounds=scenario.rounds)
agent = ModelingAgent(scenario, 'Agent', {'Teammate': FrequentistModel(scenario)})
agent_dict = {'Agent': agent, 'Teammate': teammate}

while not scenario.end(state):
    current_agent = agent_dict[state['Turn']]

    action = current_agent.get_action(state)
    new_state = scenario.transition(state, action).sample()
    print(state['Turn'], action)
    print(state)
    print(new_state)
    print('-----------------')

    for participating_agent in agent_dict.values():
        participating_agent.update(state['Turn'], state, action, new_state)

    state = new_state