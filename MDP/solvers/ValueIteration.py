from graph.State import State
from graph.StateDistribution import StateDistribution
from managers.Scenario import Scenario


def valueIteration(stateDist, scenario, horizon, gamma = 0.9):
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

    actionValues = {}
    for possibleState, possibleStateProb in stateDist.items():
        for action in scenario.getActions(possibleState):
            if action not in actionValues:
                actionValues[action] = 0

            for resultingState, resultingStateProb in scenario.transitionState(possibleState, action).items():
                if (horizon-1, resultingState) not in V:
                    value(resultingState, scenario, horizon-1, V)

                actionValues[action] += possibleStateProb * resultingStateProb * V[(horizon-1, resultingState)]

    """
    for (action, util) in actionValues.items():
        print(action, str(util))
    """

    #change such that ties are broken randomly
    return max(actionValues.items(), key=lambda x: x[1])[0]

def value(state, scenario, horizon, V, gamma = 0.9):
    # (horizon, state) is not in V; make it
    V[(horizon, state)] = scenario.getUtility(state)

    
    if horizon == 0:
        return V[(horizon, state)]

    #find max action; add expected utility
    actionValues = {}
    for action in scenario.getActions(state):
        actionValues[action] = 0
        for resultingState, stateProb in scenario.transitionState(state, action).items():
            if (horizon-1, resultingState) not in V:
                V[(horizon-1, resultingState)] = value(resultingState, scenario, horizon-1, V)

            actionValues[action] += stateProb * V[(horizon-1, resultingState)]

    #find max action, add its value to V[(horizon, state)]
    #this is where to add in action probabilities to get expectation
    maxAction = max(actionValues.items(), key= lambda x: x[1])
    V[(horizon, state)] += gamma * maxAction[1]
    return V[(horizon, state)]