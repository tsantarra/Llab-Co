from Domains.Test_Grid.GridTransitionManager import GridTransitionManager
from Domains.Test_Grid.GridActionManager import GridActionManager
from Domains.Test_Grid.GridUtilityManager import GridUtilityManager
from MDP.managers.Scenario import Scenario
from MDP.solvers.bfs import breadth_first_search
from MDP.solvers.mcts import mcts
from MDP.solvers.vi import value_iteration
from MDP.graph.State import State

def initialize():
    # Initialize managers with scenario constraints (width, height) of grid
    gtm = GridTransitionManager(25,25,5,5)
    gam = GridActionManager(25,25)
    gum = GridUtilityManager(5,5)
    return Scenario(gtm, gam, gum)

def GridTestBFS(scenario):
    # Identify goal state
    goal = State({'x':5, 'y':5})

    # Retrieve initial state.
    state = scenario.initial_state()
    print(state)

    # Plan
    plan = breadth_first_search(goal, state, scenario)

    # Execute plan
    while plan:
        action = plan.pop(0)
        state = scenario.transition_state(state, action)
        print(action)
        print(state)


def GridTestMCTS(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print(state)

    utility = 0
    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = mcts(state, scenario, 1000)
        print(node.tree_to_string(horizon=2))

        state = scenario.transition_state(state, action)
        utility += scenario.get_utility(state)

        print(action, utility)
        print(state)


def GridTestVI(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print(state)

    while not scenario.end(state):
        # Plan
        action = value_iteration(state, scenario, horizon=21)

        state = scenario.transition_state(state, action)

        print(action)
        print(state)

if __name__ == "__main__":
    # Initialize the scenario
    scenario = initialize()

    # Run tests
    print('VI Run:')
    GridTestVI(scenario)

    print('\nBFS Run:')
    GridTestBFS(scenario)

    print('\nMCTS Run:')
    GridTestMCTS(scenario)
