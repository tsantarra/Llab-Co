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
        - It does cite the LAO* approach, which is
            + Expand best partial solution
            + Create a set Z containing expanded state and ALL ancestors
            + Perform dynamic programming over set Z
            + Insight: probably have to adjust traversal to not make loops
            + Conclusion: labeling and traversal become complicated; doing a separate VI step is undesirable
        - NEW PLAN:
            + allow duplicate states
            + update with state info, not by node
            + bound horizon

"""

# TODOs
# TODO Bellman backup
# TODO Node labeling (complete/not)
# TODO Full MDP specification
# TODO Modular heuristic -> can be playouts, can be admissible evaluation function
# TODO UCT 'bias' variable = V(n)
# TODO Update? From all nodes with state? Only along traversed path?


from collections import defaultdict
from math import sqrt, log
from random import choice

from POMDP.solvers.tree.TreeNode import TreeNode


def traverse_nodes(node, scenario):
    """
    UCT down to leaf node.
    """
    while node.untried_actions == [] and len(node.children) != 0 and not scenario.end(node.state):
        # UCT to next node.
        # Perhaps just call this on node and let the tree handle the iteration.
        # Don't update state. We'll create a new state from the leaf node.
        pass

    return node


def expand_leaf(node, scenario):
    """
    Expands a new node from current leaf node.
    """

    # If conditions are met, add a leaf node.
    if node.untried_actions != [] and not scenario.end(node.state) and node.visits >= 1:
        # Randomly select move. Progress game state.
        # Want to expand nodes for all children possible
        pass

    return node


def heuristic(node, scenario):
    """
    Provides an evaluation of the state at the node given, either through playouts or by an evaluation function.
    """
    utility = 0
    state = node.state.copy()

    while not scenario.end(state):
        action = choice(scenario.actions(state))
        state = scenario.transition(state, action).sample()
        utility += scenario.utility(state)

    return utility


def backpropagate(node, utility):
    """
    Updates tree along path according to utility.
    """
    # Want to update by adjusting policy, not simply averaging the score, as in MCTS.
    pass


def tree_search(state, scenario, iterations, horizon, root_node=None, value_table=None, node_map=None):
    """
    Search game tree according to THTS.
    """
    # If a rootNode is not specified, initialize a new one.
    if not root_node:
        root_node = THTSNode(state, action_list=scenario.actions(state), is_root=True)
    passes = iterations - root_node.visits

    if not value_table:
        value_table = defaultdict(float)

    if not node_map:
        node_map = {}

    for step in range(passes):
        # Start at root
        node = root_node

        # UCT through existing nodes.
        node = traverse_nodes(node, scenario)

        # Expand a new node from leaf.
        node = expand_leaf(node, scenario, node_map)

        # Simulate rest of game randomly.
        utility = heuristic(node, scenario)

        # Backpropagate score.
        backpropagate(node, utility)

    return root_node.uct(), root_node


class THTSNode(TreeNode):
    """
    A node in the game tree.
    """
    def __init__(self, state, action=None, action_list=[], predecessors=None, is_root=False):
        """
        Initializes tree node with relevant information.
        """
        self.state = state                       # Current game state (clone of state instance).
        self.action = action                     # The move that got us to this node - "None" for the root node.
        self.untried_actions = action_list       # Yet unexplored actions
        self.is_root = is_root                   # Is this node the root node of the tree?
        self.predecessors = predecessors         # Due to the dynamic programming approach, allow multiple "parents".
        self.successors = {}                     # Action: childNode dictionary to keep links to children
        self.complete = False                    # A label to avoid sampling in complete subtrees.

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "[" + str(self.action) + " Val:" + str(self.value) + " Vis:" + str(self.visits) + "]"

    def tree_to_string(self, horizon=1, indent=0):
        """
        Builds a string representation of the tree via recursive tree traversal.
        """
        string = ''.join(['| ' for i in range(indent)]) + str(self) + '\n'
        if horizon > 0:
            for child in [node for children in self.successors.values() for node in children]:
                string += child.tree_to_string(horizon - 1, indent + 1)
        return string
