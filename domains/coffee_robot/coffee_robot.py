from domains.coffee_robot.coffee_robot_scenario import coffee_robot_scenario
from mdp.thts_dp import graph_search
from visualization.graph import show_graph


def coffee_robot(scenario):
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = graph_search(state, scenario, 1000, root_node=node)
        state = scenario.transition(state, action).sample()

        #show_graph(node, width=10, height=10)
        node = [n for n in node.successors[action] if n.state == state][0]

        print(action)
        print(state)


if __name__ == "__main__":
    coffee_robot(coffee_robot_scenario)
