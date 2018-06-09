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
from heapq import heappop, heappush
from math import sqrt, log, inf
from random import choice
from itertools import count
from functools import lru_cache
from collections import defaultdict
from copy import deepcopy, copy

from mdp.distribution import Distribution
from mdp.graph_utilities import map_graph, prune_unreachable, create_node_set, map_graph_by_depth, \
    traverse_graph_topologically


def greedy_action(node, tie_selector=choice):
    """
    Returns the action with the largest expected payoff, breaking ties randomly.
    """
    action_values = node.action_values()
    max_action, max_val = max(action_values.items(), key=lambda pair: pair[1])

    ties = [action for action in action_values if action_values[action] == max_val]
    return tie_selector(ties)


@lru_cache(maxsize=32768)
def _ucb_explore_factor(parent_visits, action_count):
    """
    Calculates the exploration term for UCT search.
    """
    return sqrt(log(parent_visits) / action_count)


def _traverse_nodes(node, tie_selector=choice):
    """
    UCT down to leaf node.
    """
    while len(node.successors) != 0:  # and not scenario.end(node.state):
        # UCT criteria with exploration factor = node value (from THTS paper)
        action_values = {act: val for act, val in node.action_values().items()
                         if act in node._incomplete_action_nodes}

        best_action, best_value = max(action_values.items(),
                                      key=lambda av: av[1] + (node.future_value + 1) *
                                      _ucb_explore_factor(node.visits, node.action_counts[av[0]]))

        tied_actions = [a for a in action_values if action_values[a] == best_value]
        selected_action = tie_selector(tied_actions)

        # Sample-based node update
        node.action_counts[selected_action] += 1
        node.visits += 1

        node = Distribution({child: prob for child, prob in node.successors[selected_action].items()
                             if not child.complete}).sample()

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
    assert not node.complete, 'ERROR. SHOULD NOT EXPAND COMPLETE NODE.'

    node.set_actions(scenario.actions(node.state))  # save computation of actions for fringe nodes

    # Expand all actions (rather than selecting one randomly)
    for action in list(node.action_space):
        transitions = scenario.transition(node.state, action)
        new_successors = Distribution()

        for new_state, prob in transitions.items():
            # Update transition utils
            node.successor_transition_values[(new_state, action)] = scenario.utility(node.state, action, new_state)

            if new_state in node_map:
                new_node = node_map[new_state]
                new_node.predecessors.add(node)
            else:
                new_node = GraphNode(state=new_state, terminal=scenario.end(new_state), predecessor=node)

                # Provide initial evaluation of new leaf node
                if not new_node.complete:
                    new_node.future_value = heuristic(new_node.state)

                # Add to node map, so we can find it later.
                node_map[new_state] = new_node

            new_successors[new_node] = prob

        # added new function call to adjust property (see GraphNode)
        node.add_new_successors(action, new_successors)

        # testing new optimization feature
        node._incomplete_action_nodes[action] = set(child for child in new_successors if not child.complete)

        # set initial action count to 1
        node.action_counts[action] = 1

    # Calculate initial action values
    node.calculate_action_values(force_recalculate=True)

    return node


def _expectation_max(node):
    """
    Sets a node's value to its immediate utility + the maximum expected utility of the
    available actions.
    """
    if node.successors:
        node.optimal_action, node.future_value = max(node.calculate_action_values().items(), key=lambda p: p[1])


def _backup(node, backup_op):
    """
    Updates tree along simulated path.
    """
    queue = [(0, node)]
    added = {node}
    post_backup_changed_nodes = [node]

    while queue:
        # Process next node
        level, node = heappop(queue)
        added.remove(node)

        # store old value for delta updates
        node._old_future_value = node.future_value

        # Value backup operator
        backup_op(node)

        # Check for changes in value
        node._has_changed = (node.future_value != node._old_future_value)

        # Update node's incomplete actions/child nodes
        for action, child_set in list(node._incomplete_action_nodes.items()):
            child_set = set(child for child in child_set if not child.complete)
            if len(child_set) == 0:
                del node._incomplete_action_nodes[action]

        # Labeling -> if all child nodes complete, this node is complete
        if not node.complete and all(child.complete for child in node.successor_set()):
            node.complete = True
            node._has_changed = True

        # Queue predecessors only if the node info has changed (policy value or complete status)
        if node._has_changed:
            post_backup_changed_nodes.append(node)
            for predecessor in node.predecessors:
                if predecessor not in added:
                    heappush(queue, (level + 1, predecessor))
                    added.add(predecessor)

    for node in post_backup_changed_nodes:
        node._has_changed = False


def search(state, scenario, iterations=inf, backup_op=_expectation_max, heuristic=None, tie_selector=choice,
           root_node=None, prune=True):
    """
    Search game tree according to THTS.
    """
    if heuristic is None:
        from functools import partial
        heuristic = partial(_rollout, scenario=scenario)

    # If a rootNode is not specified, initialize a new one.
    if root_node is None:
        root_node = GraphNode(state, scenario.end(state))

    node_map = map_graph(root_node)

    if prune:
        prune_unreachable(root_node, node_map, set())

    passes = iterations - root_node.visits + 1
    for step in count():
        # If entire tree has been searched, halt iteration.
        if root_node.complete or step > passes:
            break

        # Start at root
        node = root_node

        # UCT through existing nodes.
        node = _traverse_nodes(node, tie_selector)

        # Expand a new node from leaf.
        node = _expand_leaf(node, scenario, heuristic, node_map)

        # Recalculate state values
        _backup(node, backup_op=backup_op)

    return root_node


class GraphNode:
    """
    A node in the game tree.
    """

    def __init__(self, state, terminal, predecessor=None):
        """
        Initializes tree node with relevant information.
        """
        # State and action
        self.state = state  # Current game state (clone of state instance).
        self.scenario_end = terminal  #scenario.end(state)
        self.complete = self.scenario_end  # A label to avoid sampling in complete subgraphs.
        self.action_space = None  # scenario.actions(state)

        # Due to the dynamic programming approach, allow multiple predecessors.
        self.predecessors = {predecessor} if predecessor else set()
        self._successors = {}  # Action: child node dictionary to keep links to successors
        self._succ_set = set()
        self.successor_transition_values = {}

        # Node info for search
        self.visits = 1
        self.future_value = 0
        self.__action_values = None
        self.optimal_action = None
        self.action_counts = defaultdict(int)

        # New optimizations
        self._incomplete_action_nodes = {}
        self._has_changed = False
        self._old_future_value = self.future_value

    def set_actions(self, action_space):
        self.action_space = action_space

    def action_values(self):
        if not self.__action_values:
            return self.calculate_action_values(force_recalculate=True)
        else:
            return self.__action_values

    def calculate_action_values(self, force_recalculate=False):
        """
        For every action, return the expectation over stochastic transitions to successors states.
        """
        if not self.successors:
            return {action: 0 for action in self.action_space}

        if force_recalculate:
            self.__action_values = {
                    action: sum(probability * (self.successor_transition_values[(successor.state, action)] +
                                               successor.future_value)
                                for successor, probability in successor_distribution.items())
                    for action, successor_distribution in self.successors.items()}
        else:
            for action, succ_dist in self._successors.items():
                for child, probability in succ_dist.items():
                    if child._has_changed:
                        self.__action_values[action] += probability * (child.future_value - child._old_future_value)

        return self.__action_values

    def find_matching_successor(self, state, action=None):
        """
        Finds the successor node with matching state.
        """
        if not self.successors:
            return None

        if action:
            matches = [successor for successor in self.successors[action] if successor.state == state]
        else:
            matches = [successor for successor in self.successor_set() if successor.state == state]

        if len(matches) == 0:
            print('NO MATCHES')
        elif len(matches) > 1:
            print('MATCHING SUCCESSORS', '\n'.join(str(match) for match in matches))
        assert len(matches) == 1, '\n'.join(str(succ.state) for succ in self.successor_set()) + '\nLOOKING FOR\n' + str(
            state)
        return matches[0]

    @property
    def successors(self):
        return self._successors

    @successors.setter
    def successors(self, new_successors):
        self._successors = new_successors
        self._succ_set = set(successor for successor_dist in self.successors.values() for successor in successor_dist)

    def successor_set(self, action=None):
        """
        A short interface for grabbing child nodes without having to process the action-based successor dict.
        """
        if action:
            return self.successors[action]
        else:
            return self._succ_set

    def add_new_successors(self, action, new_successors):
        self._successors[action] = new_successors
        self._succ_set |= new_successors.keys()

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return "<" + "Val:" + "%.2f" % self.future_value + " Vis:" + str(self.visits) + ' ' + str(
            self.complete) + ">" + '\n' + str(self.state) + '\n' + str(self.state)

    def finite_horizon_string(self, horizon=1, indent=0):
        """
        Builds a string representation of the tree via recursive tree traversal.
        """
        string = ''.join(['| ' for _ in range(indent)]) + str(self) + '\n'
        if horizon > 0:
            for child_node in [node for node in self.successor_set()]:
                string += child_node.finite_horizon_string(horizon - 1, indent + 1)
        return string

    def __lt__(self, other):
        """
        Required comparison operator for queueing, etc.
        """
        return True

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        if self.state != other.state:
            return False

        return vars(self) == vars(other)

    def reachable_subgraph_size(self):
        return len(create_node_set(self))

    def copy(self):
        return copy(self)

    def copy_subgraph(self):
        # two passes -> create new nodes, then replace links between nodes
        cloned_nodes = {}

        def clone_node(node, _):
            cloned_nodes[node] = copy(node)

        def update_links(node, _):
            cloned = cloned_nodes[node]

            cloned._succ_set = set(cloned_nodes[n] for n in node._succ_set)
            cloned._successors = {action: Distribution({cloned_nodes[n]: p for n, p in dist.items()})
                                  for action, dist in node._successors.items()}

            cloned._incomplete_action_nodes = {action: set(cloned_nodes[n] for n in node_set)
                                  for action, node_set in node._incomplete_action_nodes.items()}

            cloned.predecessors = set(cloned_nodes[n] for n in node.predecessors)


        depth_map = map_graph_by_depth(self)
        traverse_graph_topologically(depth_map, clone_node)
        traverse_graph_topologically(depth_map, update_links)  # does not need to be a traversal, technically

        print('Depth Map:', len(depth_map))
        print('Cloned:', len(cloned_nodes))
        return cloned_nodes[self]

    def __deepcopy__(self, memo):
        this_id = id(self)
        if this_id in memo:
            return memo[this_id]

        cls = self.__class__
        result = cls.__new__(cls)
        memo[this_id] = result


        result.state = self.state.copy()
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
