from MDP.solvers.mcts import mcts

from MDP.Domains.Coffee_Robot.CoffeeRobotScenario import coffee_robot_scenario
from MDP.solvers.bfs import breadth_first_search
from MDP.solvers.vi import value_iteration


def coffeeRobotBFS(scenario):
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    # Plan
    plan = breadth_first_search(state, scenario)

    # Execute plan
    while plan:
        action = plan.pop(0)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)


def coffeeRobotMCTS(scenario):
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = mcts(state, scenario, 1000)
        state = scenario.transition(state, action).sample()

        # print(node.TreeToString(3))
        print(action)
        print(state)


def coffeeRobotVI(scenario):
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n',state)

    while not scenario.end(state):
        # Plan
        action = value_iteration(state, scenario, horizon=6)
        state = scenario.transition(state, action).sample()

        print(action)
        print(state)

if __name__ == "__main__":
    # Run scenario
    #coffeeRobotVI(coffee_robot_scenario)
    coffeeRobotBFS(coffee_robot_scenario)
    #coffeeRobotMCTS(coffee_robot_scenario)
