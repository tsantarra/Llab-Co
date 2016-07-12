from mdp.distribution import Distribution
from mdp.state import State
from mdp.scenario import Scenario
from mdp.graph_planner import search

from deprecated.partial_observability import po_transition, po_utility, po_actions, po_end
from domains.tiger.tiger_scenario import initial_state, actions, transition, observations, utility, end


def update_beliefs(belief_state, scenario, action, observation):
    resulting_state_distribution = Distribution()
    for state, state_prob in belief_state.items():
        for resulting_state, resulting_state_prob in scenario.transition(state, action).items():
            resulting_state_distribution[resulting_state] += state_prob * resulting_state_prob

    observation_probs = {state: scenario.observations(state, action) for state in resulting_state_distribution}
    state_probs_of_observation = {state: dist[observation] for state, dist in observation_probs.items()}

    new_belief_state = resulting_state_distribution.conditional_update(state_probs_of_observation)

    return new_belief_state


def tiger_partial_obs_test():

    scenario = Scenario(initial_state=initial_state,
                        actions=po_actions(actions),
                        transition=po_transition(transition, observations),
                        utility=po_utility(utility),
                        end=po_end(end)
                        )

    belief_state = Distribution({State({'tiger': 'Left', 'Player': 'Middle', 'Round': 0}): 0.5,
                                 State({'tiger': 'Right', 'Player': 'Middle', 'Round': 0}): 0.5})

    print('Initial beliefs:\n', belief_state)

    while not scenario.end(belief_state):
        # Plan
        action, node = search(belief_state, scenario, iterations=1000)
        belief_state = scenario.transition(belief_state, action).sample()

        print('------------------------')
        print('Action:', action)
        print('Beliefs:', belief_state)
        print('------------------------')


if __name__ == "__main__":
    tiger_partial_obs_test()
