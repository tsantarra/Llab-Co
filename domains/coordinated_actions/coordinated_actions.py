from domains.coordinated_actions.coordinated_actions_scenario import CoordinatedActionsScenario, SampledPolicyTeammate
from multiagent.modeling_agent import ModelingAgent
from multiagent.frequentist_model import FrequentistModel
from visualization.graph import show_graph
from multiagent.communication.communication_scenario import CommScenario, comm
from mdp.graph_planner import search, _greedy_action
from mdp.distribution import Distribution

# Scenario
scenario = CoordinatedActionsScenario(action_set='AB', rounds=6)
state = scenario.initial_state()

# Agents
teammate = SampledPolicyTeammate(actions=scenario.action_set, rounds=scenario.rounds)
agent = ModelingAgent(scenario, 'Agent', {'Teammate': FrequentistModel(scenario)})
agent_dict = {'Agent': agent, 'Teammate': teammate}

# Execution loop
while not scenario.end(state):
    # Have agent act
    current_agent = agent_dict[state['Turn']]
    action = current_agent.get_action(state)

    action = comm(state, agent, agent_dict)

    new_state = scenario.transition(state, action).sample()

    # Output
    print(state['Turn'], action)
    print(new_state)
    print('-----------------')

    if state['Turn'] == 'Agent':
        show_graph(agent.policy_graph_root)

    # Update agent info
    for participating_agent in agent_dict.values():
        participating_agent.update(state['Turn'], state, action, new_state)

    state = new_state
