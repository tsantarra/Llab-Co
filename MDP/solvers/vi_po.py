"""
Note: this solver has been deprecated since moving to the pure MDP approach. POMDP problems should be converted to
MDPs in belief space, then solved via the MDP solvers. The code below will fail, as scenarios no longer have the
'observations' method.
"""

from collections import defaultdict

from MDP.Distribution import Distribution


def value_iteration(belief_state, scenario, horizon, gamma=1.0):
    """
    Returns an action given the current state in a scenario.

    https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/finite-horizon-MDP.pdf
    Finite horizon:
        Keep V hashes for every horizon depth
        Build V1 from V0, V2 from V1, etc
        Vt(s) = R(s) + max ([sum([prob*Vt(resultingState) for (prob, resultingState) in transition(s, action)]) for action in getActions(s)])
        if probAction, then instead of max, do probAction * result to get expectation

    For POMDPs, the algorithm operates on distributions across states (beliefs). Steps:
        - Using possible actions, calculate a new distribution of states from state transitions.
        - For each observation, calculate a new belief distribution and pass it to the next step of the iteration.
    """
    V = {}

    utility_dist = {state: scenario.utility(state) for state in belief_state}
    V[(horizon, belief_state)] = belief_state.expectation(utility_dist)  # expectation over states now!

    if horizon == 0:
        return V[(horizon, belief_state)]

    resulting_state_distributions = defaultdict(Distribution)
    actions = scenario.actions(list(belief_state.keys())[0])  # assume available actions are identical across potential states

    # Step 1: Consider resulting state distributions for each action.
    for action in actions:
        for state, state_prob in belief_state.items():
            for resulting_state, resulting_state_prob in scenario.transition(state, action).items():
                resulting_state_distributions[action][resulting_state] += state_prob * resulting_state_prob

    action_values = defaultdict(float)
    # Step 2: Update resulting state distributions by considering how each observation would affect the belief state.
    #           - Each action/observation pair results in a new belief state.
    #           - From observation probabilities, can iterate on the value of the potential action.
    for action, state_dist in resulting_state_distributions.items():
        # Collect observation probs
        observation_probs = {state: scenario.observations(state, action) for state in state_dist}  # state -> obs probs
        possible_observations = set(key for dist in observation_probs.values() for key in dist)
        obs_probs_distributions = {obs: {state: dist[obs] for state, dist in observation_probs.items()} for obs in
                                   possible_observations}  # obs -> state probs

        # For each observation, construct new belief state, iterate
        for observation, state_probs_of_observation in obs_probs_distributions.items():
            new_belief_state = state_dist.conditional_update(state_probs_of_observation)
            if (horizon - 1, new_belief_state) not in V:
                V[(horizon - 1, new_belief_state)] = value(new_belief_state, scenario, horizon - 1, V, gamma)

            prob_of_obs = state_dist.expectation(state_probs_of_observation)
            action_values[action] += prob_of_obs * V[(horizon - 1, new_belief_state)]

    # find max action, add its value to V[(horizon, state)]
    max_action = max(action_values.items(), key=lambda x: x[1])  # tuple: (action, value)
    V[(horizon, belief_state)] += gamma * max_action[1]  # add to immediate reward (at beginning of call)
    return max_action[0]


def value(belief_state, scenario, horizon, V, gamma=1.0):
    # (horizon, state) is not in V; make it
    utility_dist = {state: scenario.utility(state) for state in belief_state}
    V[(horizon, belief_state)] = belief_state.expectation(utility_dist)  # expectation over states now!

    if horizon == 0:
        return V[(horizon, belief_state)]

    resulting_state_distributions = defaultdict(Distribution)
    actions = scenario.actions(list(belief_state.keys())[0])  # assume available actions are identical across potential states

    # Step 1: Consider resulting state distributions for each action.
    for action in actions:
        for state, state_prob in belief_state.items():
            for resulting_state, resulting_state_prob in scenario.transition(state, action).items():
                resulting_state_distributions[action][resulting_state] += state_prob * resulting_state_prob

    action_values = defaultdict(float)
    # Step 2: Update resulting state distributions by considering how each observation would affect the belief state.
    #           - Each action/observation pair results in a new belief state.
    #           - From observation probabilities, can iterate on the value of the potential action.
    for action, state_dist in resulting_state_distributions.items():
        # Collect observation probs
        observation_probs = {state: scenario.observations(state, action) for state in state_dist}  # state -> obs probs
        possible_observations = set(key for dist in observation_probs.values() for key in dist)
        obs_probs_distributions = {obs: {state: dist[obs] for state, dist in observation_probs.items()} for obs in
                                   possible_observations}  # obs -> state probs

        # For each observation, construct new belief state, iterate
        for observation, state_probs_of_observation in obs_probs_distributions.items():
            new_belief_state = state_dist.conditional_update(state_probs_of_observation)
            if (horizon - 1, new_belief_state) not in V:
                V[(horizon - 1, new_belief_state)] = value(new_belief_state, scenario, horizon - 1, V, gamma)

            prob_of_obs = state_dist.expectation(state_probs_of_observation)
            action_values[action] += prob_of_obs * V[(horizon - 1, new_belief_state)]

    # find max action, add its value to V[(horizon, state)]
    max_action = max(action_values.items(), key=lambda x: x[1], default=('None', 0))  # tuple: (action, value)
    V[(horizon, belief_state)] += gamma * max_action[1]  # add to immediate reward (at beginning of call)
    return V[(horizon, belief_state)]
