"""
This file contains a set of wrappers for the elements of MDPs, such that we can adapt them to POMDPs and retain
utilization of the generic MDP solvers. This was a proof of concept more than anything. As such, the methods
contained here are DEPRECATED.
"""


from mdp.distribution import Distribution


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

    def func(old_belief_state, action, new_belief_state):
        return new_belief_state.expectation({state: utility(old_belief_state, action, new_belief_state)
                                             for state in new_belief_state})

    return func


def po_end(end):

    def func(belief_state):
        state = next(iter(belief_state))
        return end(state)

    return func
