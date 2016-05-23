from collections import defaultdict


def value_iteration(state, scenario, horizon, gamma=1.0):
    """
    Returns an action given the current state in a scenario.

    https://web.engr.oregonstate.edu/~afern/classes/cs533/notes/finite-horizon-mdp.pdf
    Finite horizon:
        Keep V hashes for every horizon depth
        Build V1 from V0, V2 from V1, etc
        Vt(s) = R(s) + max ([sum([prob*Vt(resultingState) for (prob, resultingState) in transition(s, action)]) for action in getActions(s)])
        if probAction, then instead of max, do probAction * result to get expectation
    """
    V = {}

    action_values = defaultdict(float)
    for action in scenario.actions(state):
        for resultingState, resultingStateProb in scenario.transition(state, action).items():
            if (horizon-1, resultingState) not in V:
                value(resultingState, scenario, horizon - 1, V, gamma)

            action_values[action] += resultingStateProb * V[(horizon - 1, resultingState)]

    # change such that ties are broken randomly
    return max(action_values.items(), key=lambda x: x[1])[0]


def value(state, scenario, horizon, V, gamma=1.0):
    # (horizon, state) is not in V; make it
    V[(horizon, state)] = scenario.utility(state, None)

    if horizon == 0 or scenario.end(state):
        return V[(horizon, state)]

    # find max action; add expected utility
    action_values = defaultdict(float)
    for action in scenario.actions(state):
        for resultingState, stateProb in scenario.transition(state, action).items():
            if (horizon-1, resultingState) not in V:
                V[(horizon - 1, resultingState)] = value(resultingState, scenario, horizon - 1, V, gamma)

            action_values[action] += stateProb * V[(horizon-1, resultingState)]

    # find max action, add its value to V[(horizon, state)]
    # this is where to add in action probabilities to get expectation
    max_action = max(action_values.items(), key=lambda x: x[1])
    V[(horizon, state)] += gamma * max_action[1]
    return V[(horizon, state)]
