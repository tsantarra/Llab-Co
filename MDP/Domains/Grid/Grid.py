from MDP.solvers.mcts import mcts

from MDP.Domains.Grid.GridScenario import grid_scenario
from MDP.solvers.bfs import breadth_first_search
from MDP.solvers.vi import value_iteration


def GridTestBFS(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    # Plan
    plan = breadth_first_search(state, scenario)

    # Execute plan
    while plan:
        action = plan.pop(0)
        state = scenario.transition(state, action).sample()
        print(action)
        print(state)


def GridTestMCTS(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = mcts(state, scenario, 1000)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)


def GridTestVI(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    while not scenario.end(state):
        # Plan
        action = value_iteration(state, scenario, horizon=21)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)

if __name__ == "__main__":
    # Run tests
    print('VI Run:')
    GridTestVI(grid_scenario)

    print('\nBFS Run:')
    GridTestBFS(grid_scenario)

    print('\nMCTS Run:')
    GridTestMCTS(grid_scenario)
