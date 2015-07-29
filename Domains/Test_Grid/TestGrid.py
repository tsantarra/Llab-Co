from GridTransitionManager import GridTransitionManager
from GridActionManager import GridActionManager
from GridUtilityManager import GridUtilityManager
from managers.Scenario import Scenario
from solvers.BreadthFirstSearch import BreadthFirstSearch
from solvers.MCTS import mcts, MCTSNode
from solvers.ValueIteration import valueIteration
from graph.State import State
from graph.StateDistribution import StateDistribution

def initialize():
    #Initialize managers with scenario constraints (width, height) of grid
    gtm = GridTransitionManager(25,25,5,5)
    gam = GridActionManager(25,25)
    gum = GridUtilityManager(10,10)
    return Scenario(gtm, gam, gum)

def GridTestBFS(scenario):
    #Identify goal state
    goal = State({'x':5, 'y':5})

    #Retrieve initial state.
    state = State({'x':0, 'y':0, 'Round':0})
    print(state)

    #Plan
    plan = BreadthFirstSearch(goal, state, scenario)

    #Execute plan
    while plan:
        action = plan.pop(0)
        state = scenario.transitionState(state, action)
        print(action)
        print(state)


def GridTestMCTS(scenario):
    #Retrieve initial state.
    state = State({'x':0, 'y':0, 'Round':0})
    print(state)

    utility = 0
    node = None
    while not scenario.end(state):
        #Plan
        (action, node) = mcts(state, scenario, 1000)

        state = scenario.transitionState(state, action)
        utility += scenario.getUtility(state)

        #print(node.TreeToString(3))
        print(action, utility)
        print(state)


def GridTestVI(scenario):
    #Retrieve initial state.
    state = State({'x':0, 'y':0, 'Round':0})
    stateDist = StateDistribution(state)
    print(state)

    while not scenario.end(stateDist.collapse()):
        #Plan
        action = valueIteration(stateDist, scenario, horizon = 21)

        stateDist = scenario.transitionState(stateDist.collapse(), action)

        print(action)
        #print(stateDist.collapse())

if __name__ == "__main__":
    #Initialize the scenario
    scenario = initialize()

    GridTestVI(scenario)

    #Run tests
    print('BFS Run:')
    GridTestBFS(scenario)

    print('\nMCTS Run:')
    GridTestMCTS(scenario)



