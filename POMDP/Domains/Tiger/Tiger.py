from POMDP.Domains.Tiger.TigerScenario import tiger_scenario
from POMDP.solvers.vi import value_iteration
from MDP.State import State
from MDP.Distribution import Distribution


def update_beliefs(belief_state, scenario, action, observation):
    resulting_state_distribution = Distribution()
    for state, state_prob in belief_state.items():
        for resulting_state, resulting_state_prob in scenario.transition(state, action).items():
            resulting_state_distribution[resulting_state] += state_prob * resulting_state_prob

    observation_probs = {state: scenario.observations(state, action) for state in resulting_state_distribution}
    state_probs_of_observation = {state: dist[observation] for state, dist in observation_probs.items()}

    new_belief_state = resulting_state_distribution.conditional_update(state_probs_of_observation)

    return new_belief_state


def tigerVI(scenario):
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    belief_state = Distribution({State({'Tiger': 'Left', 'Player': 'Middle'}): 0.5,
                         State({'Tiger': 'Right', 'Player': 'Middle'}): 0.5})

    print('Initial beliefs:\n', belief_state)

    while not scenario.end(state):
        # Plan
        action = value_iteration(belief_state, scenario, horizon=10, gamma=0.9)
        state = scenario.transition(state, action).sample()
        observation = scenario.observations(state, action).sample()

        #update belief state
        belief_state = update_beliefs(belief_state, scenario, action, observation)

        print('------------------------')
        print('Action:',action)
        print('State:',state)
        print('Observation:',observation)
        print('Beliefs:',belief_state)
        print('------------------------')

if __name__ == "__main__":
    # Run scenario
    tigerVI(tiger_scenario)
