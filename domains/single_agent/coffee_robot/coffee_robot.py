from domains.single_agent.coffee_robot.coffee_robot_scenario import coffee_robot_scenario
from mdp.graph_planner import search, greedy_action


def coffee_robot(scenario):
    # Initialize state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        node = search(state, scenario, 1000, root_node=node)
        action = greedy_action(node)
        state = scenario.transition(state, action).sample()

        #show_graph(node, width=10, height=10)
        node = [n for n in node.successors[action] if n.state == state][0]

        print(action)
        print(state)


if __name__ == "__main__":
    coffee_robot(coffee_robot_scenario)
