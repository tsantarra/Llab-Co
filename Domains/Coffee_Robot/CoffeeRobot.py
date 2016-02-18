from Domains.Coffee_Robot.CoffeeRobotTransitionManager import CoffeeRobotTransitionManager
from Domains.Coffee_Robot.CoffeeRobotActionManager import CoffeeRobotActionManager
from Domains.Coffee_Robot.CoffeeRobotUtilityManager import CoffeeRobotUtilityManager

from MDP.solvers.bfs import breadth_first_search
from MDP.solvers.mcts import mcts, MCTSNode
from MDP.solvers.vi import value_iteration

from MDP.graph.State import State

from MDP.managers.Scenario import Scenario

def initialize():
    # Initialize managers
    crtm = CoffeeRobotTransitionManager()
    cram = CoffeeRobotActionManager()
    crum = CoffeeRobotUtilityManager()
    return Scenario(crtm, cram, crum)

def coffeeRobotBFS(scenario):
    # Identify goal state
    goal = State({'H':True, 'C':False, 'W':False, 'R':True, 'U':True, 'O':True })

    # Initialize state.
    state = State({'H':False, 'C':False, 'W':False, 'R':True, 'U':False, 'O':True, 'Round':0})
    print(state)

    # Plan
    plan = breadth_first_search(goal, state, scenario)

    # Execute plan
    while plan:
        action = plan.pop(0)
        state = scenario.transition_state(state, action)
        print(action)
        print(state)

def coffeeRobotMCTS(scenario):
    # Initialize state.
    state = State({'H':False, 'C':False, 'W':False, 'R':True, 'U':False, 'O':True, 'Round':0})
    print('Initial state:',state)

    utility = 0
    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = mcts(state, scenario, 1000)

        state = scenario.transition_state(state, action)
        utility += scenario.get_utility(state)

        # print(node.TreeToString(3))
        print(action, utility)
        print(state)

def coffeeRobotVI(scenario):
    # Initialize state.
    state = State({'H':False, 'C':False, 'W':False, 'R':True, 'U':False, 'O':True, 'Round':0})
    print(state)

    while not scenario.end(state):
        # Plan
        action = value_iteration(state, scenario, horizon = 6)

        state = scenario.transition_state(state, action)

        print(action)
        print(state)

if __name__ == "__main__":
    # Initialize and run scenario
    scenario = initialize()
    coffeeRobotVI(scenario)
    # coffeeRobotBFS(scenario)
    # coffeeRobotMCTS(scenario)
