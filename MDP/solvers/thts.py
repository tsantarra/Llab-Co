"""
    Rather than implement normal UCT, we take elements from THTS as we are working with MDPs.

    Paper: Trial-based Heuristic Tree Search for Finite Horizon MDPs
    Link: http://www2.informatik.uni-freiburg.de/~ki/papers/keller-helmert-icaps2013.pdf

    Differences:
        - Partial Bellman backups. V(n) = R(n) + (sum(P(c|n) * V(c)) / (sum(P(c|n))
          where c are explored children (i.e. denominator may not equal 1)
        - Value is no longer average over children but maximum choice, reflecting policy choices.
        - Solve label? Conditions: goal state OR all children are solved.
            + if so, never choose solved node
        - Cut off trial at expansion; do not simulate remainder? -> only if admissible heuristic
"""

from math import sqrt, log
from random import choice

from MDP.solvers.tree.TreeNode import TreeNode
from MDP.Distribution import Distribution


def traverse_nodes(node, scenario):
    """
    UCT down to leaf node.
    """
    while node.untried_actions == [] and len(node.children) != 0 and not scenario.end(node.state):
        # Get UCT action. Progress game state.
        node = node.uct()

    return node


def expand_leaf(node, scenario):
    """
    Expands a new node from current leaf node.
    """

    # If conditions are met, add a leaf node.
    if node.untried_actions != [] and not scenario.end(node.state) and node.visits >= 1:
        # Randomly select move. Progress game state.
        action = choice(node.untried_actions)

        # Add new nodes to tree.
        node.add_child(node.state, action, scenario)
        node = node.children[action].sample()
    return node


def heuristic(node, scenario):
    """
    Plays remainder of game randomly, accruing utility.
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
    Updates tree along path according to reward.
    """
    while node is not None:
        node.update(utility)
        node = node.parent


def mcts(state, scenario, iterations, root_node=None):
    """
    Search game tree according to MCTS. 
    """
    # If a rootNode is not specified, initialize a new one.
    if not root_node:
        root_node = MCTSNode(state, action_list=scenario.actions(state), is_root=True)
    passes = iterations - root_node.visits

    for step in range(passes):
        # Start at root
        node = root_node

        # UCT through existing nodes.
        node = traverse_nodes(node, scenario)

        # Expand a new node from leaf.
        node = expand_leaf(node, scenario)

        # Simulate rest of game randomly.
        utility = heuristic(node, scenario)

        # Backpropagate score.
        backpropagate(node, utility)

    return root_node.uct(), root_node


class MCTSNode(TreeNode):
    """
    A node in the game tree. 
    """

    def __init__(self, state, action=None, action_list=None, parent=None, is_root=False):
        """
        Initializes tree node with relevant information.
        """
        super().__init__(parent, children=None)
        if action_list is None:
            action_list = []

        self.state = state  # Current game state (clone of game instance).
        self.action = action  # The move that got us to this node - "None" for the root node.
        self.untried_actions = action_list  # Yet unexplored actions
        self.parent = parent  # Parent node to this node
        self.is_root = is_root  # Is this node the root node of the tree?

        self.children = {}  # Action: childNode dictionary to keep links to children
        self.value = 0.0  # Average score of all paths through this node. TODO Bellman backup/DP approach
        self.visits = 0  # Number of times this node has been visited.

    def uct(self, explore_factor=1):
        """
        Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
        lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
        exploration versus exploitation.
        """
        action_values = {}
        action_counts = {}
        for action in self.children:
            action_values[action] = sum(c.value * self.children[action][c] for c in self.children[action])
            action_values[action] /= sum(self.children[action].values())
            action_counts[action] = sum(c.visits for c in self.children[action])

        best_action = max(action_values, lambda a: action_values[a]
                                                   + explore_factor * sqrt(2 * log(self.visits)/action_counts[a]))
        best_val = action_values[best_action]
        tied_actions = [a for a in action_values if action_values[a] == best_val]

        return self.children[choice(tied_actions)].sample()

    def add_child(self, state, action, scenario):
        """
        Remove action from untried_actions and add a new child nodes for possible transitions.
        """
        if action in self.untried_actions:
            self.untried_actions.remove(action)

        transitions = scenario.transition(state, action)
        new_children = Distribution({MCTSNode(new_state, action, scenario.actions(new_state), parent=self): prob
                                     for new_state, prob in transitions.items()})

        self.children[action] = new_children

    def update(self, utility):
        """
        Update this node - one additional visit and result additional wins.
        """
        self.visits += 1
        # TODO
        # also, need to figure out where and how utility is stored. In the node? In the backup ref? What's kept here?
        ####self.value = (self.value * (self.visits - 1) + utility) / self.visits # TODO CONVERT TO POLICY CALC

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "[" + str(self.action) + " Val:" + str(self.value) + " Vis:" + str(self.visits) + "]"

    def tree_to_string(self, horizon=1, indent=0):
        """
        Builds a string representation of the tree via recursive tree traversal.
        """
        string = ''.join(['| ' for _ in range(indent)]) + str(self) + '\n'
        if horizon > 0:
            for child in [node for children in self.children.values() for node in children]:
                string += child.tree_to_string(horizon - 1, indent + 1)
        return string
