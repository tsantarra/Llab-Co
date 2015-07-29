from MDP.graph.State import State
from MDP.managers.Scenario import Scenario

def BreadthFirstSearch(goal, initialState, scenario):
    """Searches for the goal state using BFS."""
    #Copy intial state to work with.
    state = initialState.copy()

    #Initialize queue of state/plan pairs.
    queue = [(state, [])]

    #States covered in search
    statesCovered = set()

    #While states remain to be explored, search the state/plan queue.
    while queue: 
        #Retrieve state/plan to examine.
        currentState, plan = queue.pop(0) 

        #If goal reached, return plan.
        #Ignore state variables not relevant to goal.
        goalCheck = State({key:currentState[key] for key in goal.keys()})
        if goalCheck == goal:
            return plan

        #Expand state by one step. Add resulting state/plan pairs to queue.
        for action in scenario.getActions(currentState):
            #Get new state after action
            newState = scenario.transitionState(currentState, action)

            #Convert state to hashable representation. Check if search has already covered said state.
            stateRep = tuple(newState.items())
            if stateRep not in statesCovered:
                #If not yet reached, add to covered states, adjust plan, and add to queue.
                statesCovered.add(stateRep)
                newPlan = list(plan) + [action]
                queue += [(newState, newPlan)] 

    #No successful plan found
    return None