"""
For simplicity, we will treat a Scenario as a collection of functions which operate on States.
"""
from MDP.Distribution import Distribution
from collections import namedtuple

Scenario = namedtuple('Scenario', ['initial_state', 'transition', 'actions', 'utility', 'end'])
Scenario.__new__.__defaults__ = (None,) * len(Scenario._fields)  # Sets default values to None


def po_transition(transition, observations):

    def func(belief_state, action):
        # Calculate expected distribution of states given belief state and taken action
        resulting_state_dist = Distribution()
        for state, state_prob in belief_state.items():
            for resulting_state, resulting_state_prob in transition(state, action).items():
                resulting_state_dist[resulting_state] += state_prob * resulting_state_prob

        # Collect observation probs
        observation_probs = {state: observations(state, action) for state in resulting_state_dist}
        possible_observations = set(key for dist in observation_probs.values() for key in dist)
        obs_probs_distributions = {obs: {state: dist[obs] for state, dist in observation_probs.items()} for obs in
                                   possible_observations}  # obs -> state probs

        # Transition is to return a set of resulting states associated with stochastic probabilities.
        # Here, we return a set of new belief states associated with probabilities of the observations causing them.
        possible_new_belief_states = Distribution()
        for observation, state_probs_of_observation in obs_probs_distributions.items():
            new_belief_state = resulting_state_dist.conditional_update(state_probs_of_observation)

            prob_of_obs = resulting_state_dist.expectation(state_probs_of_observation)  # total prob of this observation
            possible_new_belief_states[new_belief_state] = prob_of_obs

        return possible_new_belief_states

    return func


def po_actions(actions):

    def func(belief_state):
        state = next(iter(belief_state))
        return actions(state)

    return func


def po_utility(utility):

    def func(belief_state, action):
        return belief_state.expectation({state: utility(state) for state in belief_state})

    return func


def po_end(end):

    def func(belief_state):
        state = next(iter(belief_state))
        return end(state)

    return func
