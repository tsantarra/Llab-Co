import mdp.graph_planner as dpthts


class SampledPolicyTeammate:
    pass



"""
alternate communication strategies
    - EARLY TERMINATION CONDITION
    - exhaustive (small domain only?)
    - most disagreed on
    - most variance?
    - product of the two?
    - myopic
"""


def calculate_optimal_policy(scenario):
    # Retrieve initial state.
    state = scenario.initial_state()
    print('Initial state:\n', state)

    node = None
    while not scenario.end(state):
        # Plan
        (action, node) = dpthts.search(state, scenario, 1000, root_node=node)
        state = scenario.transition(state, action).sample()

        node = [node for node in node.successors[action] if node.state == state][0]

        print(action)
        print(state)


def make_teammates():
    policy = calculate_optimal_policy()

    # do stuff to sample

