from MDP.Distribution import Distribution
from MDP.State import State


def update_beliefs(belief_state, scenario, action, observation):
    resulting_state_distribution = Distribution()
    for state, state_prob in belief_state.items():
        for resulting_state, resulting_state_prob in scenario.transition(state, action).items():
            resulting_state_distribution[resulting_state] += state_prob * resulting_state_prob

    observation_probs = {state: scenario.observations(state, action) for state in resulting_state_distribution}
    state_probs_of_observation = {state: dist[observation] for state, dist in observation_probs.items()}

    new_belief_state = resulting_state_distribution.conditional_update(state_probs_of_observation)

    return new_belief_state


def tigerPOVI(scenario):
    from MDP.solvers.vi_po import value_iteration
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    belief_state = Distribution({State({'Tiger': 'Left', 'Player': 'Middle', 'Round': 0}): 0.5,
                         State({'Tiger': 'Right', 'Player': 'Middle', 'Round': 0}): 0.5})

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


def tigerWrappedVI():
    from MDP.Scenario import Scenario
    from MDP.Scenario import po_transition, po_utility, po_actions, po_end
    from Domains.Tiger.TigerScenario import initial_state, actions, transition, observations, utility, end
    from MDP.solvers.vi import value_iteration

    scenario = Scenario(initial_state=initial_state,
                        actions=po_actions(actions),
                        transition=po_transition(transition, observations),
                        utility=po_utility(utility),
                        end=po_end(end)
                        )

    belief_state = Distribution({State({'Tiger': 'Left', 'Player': 'Middle', 'Round': 0}): 0.5,
                                 State({'Tiger': 'Right', 'Player': 'Middle', 'Round': 0}): 0.5})

    print('Initial beliefs:\n', belief_state)

    while not scenario.end(belief_state):
        # Plan
        action = value_iteration(belief_state, scenario, horizon=10, gamma=0.9)
        belief_state = scenario.transition(belief_state, action).sample()

        print('------------------------')
        print('Action:', action)
        print('Beliefs:', belief_state)
        print('------------------------')


def tigerWrappedTHTS():
    from MDP.Scenario import Scenario
    from MDP.Scenario import po_transition, po_utility, po_actions, po_end
    from Domains.Tiger.TigerScenario import initial_state, actions, transition, observations, utility, end
    from MDP.solvers.thts_dp import tree_search

    scenario = Scenario(initial_state=initial_state,
                        actions=po_actions(actions),
                        transition=po_transition(transition, observations),
                        utility=po_utility(utility),
                        end=po_end(end)
                        )

    belief_state = Distribution({State({'Tiger': 'Left', 'Player': 'Middle', 'Round': 0}): 0.5,
                                 State({'Tiger': 'Right', 'Player': 'Middle', 'Round': 0}): 0.5})

    print('Initial beliefs:\n', belief_state)

    while not scenario.end(belief_state):
        # Plan
        action, node = tree_search(belief_state, scenario, iterations=1000)
        belief_state = scenario.transition(belief_state, action).sample()

        print('------------------------')
        print('Action:', action)
        print('Beliefs:', belief_state)
        print('------------------------')

if __name__ == "__main__":
    # Run scenario
    #tigerPOVI(tiger_scenario)
    #tigerWrappedVI()
    tigerWrappedTHTS()
