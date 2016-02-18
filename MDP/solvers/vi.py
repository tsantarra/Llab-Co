

def value_iteration(state, scenario, horizon, gamma = 0.9):
    """
    Returns an action given the current state in a scenario.

    https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/finite-horizon-MDP.pdf
    Finite horizon:
        Keep V hashes for every horizon depth
        Build V1 from V0, V2 from V1, etc
        Vt(s) = R(s) + max ([sum([prob*Vt(resultingState) for (prob, resultingState) in transition(s, action)]) for action in getActions(s)])
        if probAction, then instead of max, do probAction * result to get expectation
    """
    V = {}

    action_values = {}
    for action in scenario.get_actions(state):
        if action not in action_values:
            action_values[action] = 0

        for resultingState, resultingStateProb in scenario.transition_state(state, action, all_outcomes=True):
            if (horizon-1, resultingState) not in V:
                value(resultingState, scenario, horizon-1, V)

            action_values[action] +=  resultingStateProb * V[(horizon-1, resultingState)]

    # change such that ties are broken randomly
    return max(action_values.items(), key=lambda x: x[1])[0]


def value(state, scenario, horizon, V, gamma = 0.9):
    # (horizon, state) is not in V; make it
    V[(horizon, state)] = scenario.get_utility(state)

    if horizon == 0:
        return V[(horizon, state)]

    # find max action; add expected utility
    action_values = {}
    for action in scenario.get_actions(state):
        action_values[action] = 0
        for resultingState, stateProb in scenario.transition_state(state, action, all_outcomes=True):
            if (horizon-1, resultingState) not in V:
                V[(horizon-1, resultingState)] = value(resultingState, scenario, horizon-1, V)

            action_values[action] += stateProb * V[(horizon-1, resultingState)]

    # find max action, add its value to V[(horizon, state)]
    # this is where to add in action probabilities to get expectation
    max_action = max(action_values.items(), key= lambda x: x[1])
    V[(horizon, state)] += gamma * max_action[1]
    return V[(horizon, state)]