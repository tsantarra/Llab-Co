from Domains.Coffee_Robot.CoffeeRobotTransitionManager import CoffeeRobotTransitionManager
from Domains.Coffee_Robot.CoffeeRobotActionManager import CoffeeRobotActionManager
from Domains.Coffee_Robot.CoffeeRobotUtilityManager import CoffeeRobotUtilityManager

from MDP.solvers.BreadthFirstSearch import BreadthFirstSearch
from MDP.solvers.MCTS import mcts, MCTSNode
from MDP.solvers.ValueIteration import valueIteration

from MDP.graph.State import State
from MDP.graph.StateDistribution import StateDistribution

from MDP.managers.Scenario import Scenario

def initialize():
    #Initialize managers 
    crtm = CoffeeRobotTransitionManager()
    cram = CoffeeRobotActionManager()
    crum = CoffeeRobotUtilityManager()
    return Scenario(crtm, cram, crum)

def coffeeRobotBFS(scenario):
    #Identify goal state
    goal = State({'H':True, 'C':False, 'W':False, 'R':True, 'U':True, 'O':True})

    #Initialize state.
    state = State({'H':False, 'C':False, 'W':False, 'R':True, 'U':False, 'O':True})
    print(state)

    #Plan
    plan = BreadthFirstSearch(goal, state, cram, crtm)

    #Execute plan
    while plan:
        action = plan.pop(0)
        state = crtm.transition(state, action)
        print(action)
        print(state)

def coffeeRobotMCTS(scenario):
    #Initialize state.
    state = State({'H':False, 'C':False, 'W':False, 'R':True, 'U':False, 'O':True, 'Round':0})
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

def coffeeRobotVI(scenario):
    #Initialize state.
    state = State({'H':False, 'C':False, 'W':False, 'R':True, 'U':False, 'O':True, 'Round':0})
    stateDist = StateDistribution(state)
    print(state)

    while not scenario.end(stateDist.collapse()):
        #Plan
        action = valueIteration(stateDist, scenario, horizon = 6)

        stateDist = scenario.transitionState(stateDist.collapse(), action)

        print(action)
        print(stateDist.collapse())

if __name__ == "__main__":
    #Initialize and run scenario
    scenario = initialize()
    #coffeeRobotVI(scenario)
    #coffeeRobotBFS(scenario)
    coffeeRobotMCTS(scenario)
