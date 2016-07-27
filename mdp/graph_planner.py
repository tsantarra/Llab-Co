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

import logging
from heapq import heappop, heappush
from math import sqrt, log
from random import choice

from mdp.distribution import Distribution


def _greedy_action(node):
    """
    UCT to next node.
    """
    action_values = {}
    action_counts = {}
    for action in node.successors:
        action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
        action_values[action] /= sum(node.successors[action].values())
        action_counts[action] = sum(child.visits for child in node.successors[action])

    best_action, best_value = max(action_values.items(), key=lambda av: av[1])

    tied_actions = [a for a, v in action_values.items() if v == best_value]

    return max(tied_actions, key=lambda a: action_counts[a])


def _traverse_nodes(node, scenario):
    """
    UCT down to leaf node.
    """
    while node.untried_actions == [] and len(node.successors) != 0:  # and not scenario.end(node.state):
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


def _rollout(state, scenario):
    """
    Provides an evaluation of the state at the node given, either through playouts or by an evaluation function.
    """
    utility = 0

    while not scenario.end(state):
        action = choice(scenario.actions(state))
        new_state = scenario.transition(state, action).sample()
        utility += scenario.utility(state, action, new_state)
        state = new_state

    return utility


def _expand_leaf(node, scenario, heuristic, node_map):
    """
    Expands one or more new nodes from current leaf node.
    """

    # If conditions are met, add one or more leaf nodes.
    if node.untried_actions != [] and not node.complete:  # scenario.end(node.state):
        # Expand all actions (rather than selecting one randomly)
        expand_actions = node.untried_actions
        node.untried_actions = []
        for action in expand_actions:
            transitions = scenario.transition(node.state, action)
            new_successors = Distribution()
            for new_state, prob in transitions.items():
                if new_state in node_map:
                    new_node = node_map[new_state]
                    new_node.predecessors.add(node)
                else:
                    new_node = GraphNode(state=new_state, scenario=scenario, action=action, predecessor=node)

                    # Provide initial evaluation of new leaf node
                    new_node.value = new_node.immediate_value
                    if not new_node.complete:  # not scenario.end(new_node.state):
                        new_node.value += heuristic(new_node.state)

                    # Add to node map, so we can find it later.
                    node_map[new_state] = new_node

                new_successors[new_node] = prob

            node.successors[action] = new_successors

    return node


def _expectation_max(node):
    """
    Sets a node's value to its immediate utility + the maximum expected utility of the
    available actions.
    """
    if node.successors:
        action_values = {}
        for action in node.successors:
            action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
            action_values[action] /= sum(node.successors[action].values())

        node.value = node.immediate_value + max(action_values.values())


def _backup(node, scenario, backup_op):
    """
    Updates tree along simulated path.
    """
    queue = [(0, node)]
    added = {node}
    while queue:
        # Process next node
        level, node = heappop(queue)
        added.remove(node)

        # Sample-based node update
        node.visits += 1

        # Value backup operator
        backup_op(node)

        # Labeling -> if all actions expanded and all child nodes complete, this node is complete
        # Scenario end checked in constructor. This is redundant.
        # if not node.complete and scenario.end(node.state):
        #    node.complete = True
        if not node.complete and node.untried_actions == [] and \
                all(child.complete for child_set in node.successors.values() for child in child_set):
            node.complete = True

        # Queue predecessors
        for predecessor in node.predecessors:
            if predecessor not in added:
                heappush(queue, (level + 1, predecessor))
                added.add(predecessor)


def create_node_set(node, node_set):
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    node_set.add(node)
    for successor in [successor_dist for dist in node.successors.values()
                      for successor_dist in dist if successor_dist.state not in node_set]:
        create_node_set(successor, node_set)


def detect_cycle(node):
    """
    Uses DFS to detect if there exist any cycles in the directed graph.
    """
    # Define recursive depth-first search function
    def dfs(current, unv, inc, comp):
        unv.remove(current)
        inc.add(current)

        for successor in [successor_dist for dist in current.successors.values()
                          for successor_dist in dist]:
            if successor in comp:
                continue
            elif successor in inc:
                return True
            elif dfs(successor, unv, inc, comp):
                return True

        inc.remove(current)
        comp.add(current)
        return False

    # Create initial sets
    unvisited, incomplete, complete = set(), set(), set()
    create_node_set(node, unvisited)

    # Return result (all nodes are reachable from the node, so only one call is necessary)
    return dfs(node, unvisited, incomplete, complete)


def map_tree(node):
    """
    Builds a dict mapping of states to corresponding nodes in the graph.
    """
    node_set = set()
    create_node_set(node, node_set)
    return {n.state: n for n in node_set}


def _prune(node, node_map, checked):
    """
    Prunes currently unreachable nodes from the graph, which cuts down on policy computation time for
    irrelevant areas of the state space.
    """
    checked.add(node)
    node.predecessors = set(pred for pred in node.predecessors if pred.state in node_map)
    for successor_dist in node.successors.values():
        for successor in [succ for succ in successor_dist if succ not in checked]:
            _prune(successor, node_map, checked)


def search(state, scenario, iterations, backup_op=_expectation_max, heuristic=_rollout, root_node=None):
    """
    Search game tree according to THTS.
    """
    # If a rootNode is not specified, initialize a new one.
    if root_node is None:
        root_node = GraphNode(state, scenario)
        node_map = map_tree(root_node)
    else:
        node_map = map_tree(root_node)
        _prune(root_node, node_map, set())

    passes = iterations - root_node.visits + 1
    for step in range(passes):
        # If entire tree has been searched, halt iteration.
        if root_node.complete:
            break

        # Start at root
        node = root_node

        # UCT through existing nodes.
        node = _traverse_nodes(node, scenario)

        # Expand a new node from leaf.
        node = _expand_leaf(node, scenario, heuristic, node_map)

        # Recalculate state values
        _backup(node, scenario, backup_op=backup_op)

    return _greedy_action(root_node), root_node


class GraphNode:
    """
    A node in the game tree.
    """

    def __init__(self, state, scenario, action=None, predecessor=None):
        """
        Initializes tree node with relevant information.
        """
        self.state = state  # Current game state (clone of state instance).
        self.untried_actions = scenario.actions(state)  # Yet unexplored actions
        self.complete = scenario.end(state)  # A label to avoid sampling in complete subtrees.

        # Due to the dynamic programming approach, allow multiple "parents".
        self.predecessors = {predecessor} if predecessor else set()
        self.successors = {}  # Action: childNode dictionary to keep links to successors

        self.visits = 1  # visits of the node so far; count initialization as a visit
        self.immediate_value = scenario.utility(predecessor.state if predecessor else None, action, state)
        self.value = 0  # total immediate value + future value

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "<" + "Val:" + "%.2f" % self.value + " Vis:" + str(self.visits) + ">"  # + '\n' + str(self.state)

    def finite_horizon_string(self, horizon=1, indent=0):
        """
        Builds a string representation of the tree via recursive tree traversal.
        """
        string = ''.join(['| ' for _ in range(indent)]) + str(self) + '\n'
        if horizon > 0:
            for child_node in [node for successors in self.successors.values() for node in successors]:
                string += child_node.finite_horizon_string(horizon - 1, indent + 1)
        return string

    def __lt__(self, other):
        """
        Required comparison operator for queueing, etc.
        """
        return True
