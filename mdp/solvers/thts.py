"""
    Adaptation of MCTS to MDPs, incorporating full domain knowledge (primarily transition probabilities) and dynamic
    programming backups.

    Original paper: Trial-based Heuristic Tree Search for Finite Horizon MDPs
    Link: https://fai.cs.uni-saarland.de/teaching/summer-15/mdp-material/area3-topic3-finite-horizon.pdf

    Modifications:
        - Bellman backup
        - Node labeling (complete/not)
        - Full MDP specification
        - Modular heuristic -> can be playouts, can be admissible evaluation function
        - UCT 'bias' variable = V(n)

    Note: It is unclear how to incorporate a discount factor without necessitating the derivation of state values at
    each possible horizon depth. Therefore, this implementation omits horizon reward discounting. This should be
    considered in any comparison to the value iteration implementation, which may discount rewards.

    CYCLES
        - The THTS paper does not explain how to handle cycles well, other than cap the horizon depth.
        - It does cite the LAO* approach
            + Expand best partial solution
            + Create a set Z containing expanded state and ALL ancestors
            + Perform dynamic programming over set Z
            + Insight: adjust traversal to not make loops

    Assuming non-cyclic, relevant papers:
        - UCD: Upper Confidence bound for rooted Directed acyclic graphs
        - Transpositions and Move Groups in Monte Carlo Tree Search

"""

from math import sqrt, log
from random import choice

from mdp.distribution import Distribution


def greedy_action(node):
    """
    UCT to next node.
    """
    action_values = {}
    for action in node.successors:
        action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
        action_values[action] /= sum(node.successors[action].values())

    best_action, best_value = max(action_values.items(), key=lambda av: av[1])

    tied_actions = [a for a, v in action_values.items() if v == best_value]

    return choice(tied_actions)


def traverse_nodes(node, scenario):
    """
    UCT down to leaf node.
    """
    while node.untried_actions == [] and len(node.successors) != 0 and not scenario.end(node.state):
        # With node labeling (complete), we only want to consider incomplete successors.
        incomplete_successors = {act: {child: prob for child, prob in node.successors[act].items()
                                       if not child.complete}
                                 for act in node.successors}
        assert not all(len(successor_dist) == 0 for successor_dist in incomplete_successors.values()), \
            'No legal targets for traversal.' + '\n' + node.tree_to_string(horizon=1)

        action_values = {}
        action_counts = {}
        for action in incomplete_successors:
            if len(incomplete_successors[action]) == 0:
                continue
            action_values[action] = sum(child.value * prob for child, prob in incomplete_successors[action].items())
            action_values[action] /= sum(incomplete_successors[action].values())
            action_counts[action] = sum(child.visits for child in incomplete_successors[action])

        # UCT criteria with exploration factor = node value (from THTS paper)
        best_action, best_value = max(action_values.items(),
                                      key=lambda av: av[1] +
                                                     (node.value + 1) * sqrt(log(node.visits) / action_counts[av[0]]))

        tied_actions = [a for a in action_values if action_values[a] == best_value]

        node = Distribution(incomplete_successors[choice(tied_actions)]).sample()

    return node


def rollout(node, scenario):
    """
    Provides an evaluation of the state at the node given, either through playouts or by an evaluation function.
    """
    utility = 0
    state = node.state.copy()

    while not scenario.end(state):
        action = choice(scenario.actions(state))
        state = scenario.transition(state, action).sample()
        utility += scenario.utility(state, action)

    return utility


def expand_leaf(node, scenario, heuristic):
    """
    Expands a new node from current leaf node.
    """

    # If conditions are met, add a leaf node.
    if node.untried_actions != [] and not scenario.end(node.state):
        # Randomly select move. Progress game state.
        action = choice(node.untried_actions)
        node.untried_actions.remove(action)

        transitions = scenario.transition(node.state, action)
        new_successors = Distribution({THTSNode(new_state, scenario, action=action, predecessor=node): prob
                                       for new_state, prob in transitions.items()})

        # Provide initial heuristic evaluation of leaf
        for successor in new_successors:
            successor.value = successor.immediate_value + heuristic(successor, scenario)

        node.successors[action] = new_successors

    return node


def backup(node, scenario):
    """
    Updates tree along simulated path.
    """
    while node:
        node.visits += 1

        if node.successors:
            action_values = {}
            for action in node.successors:
                action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
                action_values[action] /= sum(node.successors[action].values())

            node.value = node.immediate_value + max(action_values.values())

        # Labeling -> if all actions expanded and all child nodes complete, this node is complete
        if scenario.end(node.state):
            node.complete = True
        elif node.untried_actions == [] and \
                all(child.complete for child_set in node.successors.values() for child in child_set):
            node.complete = True

        node = node.predecessor


def tree_search(state, scenario, iterations, heuristic=rollout, root_node=None):
    """
    Search game tree according to THTS.
    """
    # If a rootNode is not specified, initialize a new one.
    if not root_node:
        root_node = THTSNode(state, scenario)
    passes = iterations - root_node.visits + 1

    for step in range(passes):
        if root_node.complete:
            break

        # Start at root
        node = root_node

        # UCT through existing nodes.
        node = traverse_nodes(node, scenario)

        # Expand a new node from leaf.
        node = expand_leaf(node, scenario, heuristic)

        # Recalculate state values
        backup(node, scenario)

    return greedy_action(root_node), root_node


class THTSNode:
    """
    A node in the game tree.
    """

    def __init__(self, state, scenario, action=None, predecessor=None):
        """
        Initializes tree node with relevant information.
        """
        self.state = state  # Current game state (clone of state instance).
        self.action = action  # The move that got us to this node - "None" for the root node.
        self.untried_actions = scenario.actions(state)  # Yet unexplored actions
        self.complete = False  # A label to avoid sampling in complete subtrees.

        self.predecessor = predecessor  # Due to the dynamic programming approach, allow multiple "parents".
        self.successors = {}  # Action: childNode dictionary to keep links to successors

        self.visits = 1  # visits of the node so far; count initialization as a visit
        self.immediate_value = scenario.utility(state, action)
        self.value = 0  # total immediate value + future value

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "[" + str(self.action) + " Val:" + str(self.value) + " Vis:" + str(self.visits) + "] " + str(self.complete)

    def tree_to_string(self, horizon=1, indent=0):
        """
        Builds a string representation of the tree via recursive tree traversal.
        """
        string = ''.join(['| ' for _ in range(indent)]) + str(self) + '\n'
        if horizon > 0:
            for child_node in [node for successors in self.successors.values() for node in successors]:
                string += child_node.tree_to_string(horizon - 1, indent + 1)
        return string

    def unique_nodes(self, seen):
        seen.add(self)
        for child_node in [node for successors in self.successors.values() for node in successors]:
            child_node.unique_nodes(seen)

    def __len__(self):
        unique = set()
        self.unique_nodes(unique)
        return len(unique)
