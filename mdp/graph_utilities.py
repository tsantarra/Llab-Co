from collections import defaultdict, deque
from operator import mul
from functools import reduce

from mdp.state import State


def recursive_traverse_policy_graph(node, node_values, model_state, policy, policy_fn, agent_identity):
    """ Computes a policy maximizing expected utility given a probabilistic agent model for other agents. """
    # Already visited this node. Return its computed value.
    if node in node_values:
        return node_values[node]

    # Leaf node. Value already calculated as immediate + heuristic.
    if not node.successors:
        node_values[node] = node.future_value
        return node.future_value

    # Update all individual agent models
    individual_agent_actions = node.action_space.individual_actions()
    world_state = node.state['World State']
    resulting_models = {agent_name: {action: model_state[agent_name].update(world_state, action) for action in
                                     agent_actions}
                        for agent_name, agent_actions in individual_agent_actions.items()
                        if agent_name in model_state}

    # Calculate expected return for each action at the given node
    joint_action_values = defaultdict(float)
    for joint_action, result_distribution in node.successors.items():
        assert abs(sum(result_distribution.values()) - 1.0) < 10e-5, 'Action probabilities do not sum to 1.'

        # Construct new model state from individual agent models
        new_model_state = State({agent_name: resulting_models[agent_name][joint_action[agent_name]]
                                 for agent_name in model_state})

        # Traverse to successor nodes
        for resulting_state_node, result_probability in result_distribution.items():
            resulting_node_value = recursive_traverse_policy_graph(resulting_state_node, node_values, new_model_state,
                                                                   policy, policy_fn, agent_identity)

            joint_action_values[joint_action] += result_probability * (node.successor_transition_values[
                                                                           (resulting_state_node.state, joint_action)] +
                                                                       resulting_node_value)

    # Now breakdown joint actions so we can calculate the primary agent's action values
    agent_individual_actions = node.action_space.individual_actions()
    other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                               for other_agent, other_agent_model in model_state.items()}

    agent_action_values = {action: 0 for action in agent_individual_actions[agent_identity]}
    for agent_action in agent_individual_actions[agent_identity]:
        all_joint_actions_with_fixed_action = [action for action in joint_action_values
                                               if action[agent_identity] == agent_action]

        for joint_action in all_joint_actions_with_fixed_action:
            probability_of_action = reduce(mul, [other_agent_action_dist[joint_action[other_agent]]
                                                 for other_agent, other_agent_action_dist in
                                                 other_agent_predictions.items()])
            agent_action_values[agent_action] += probability_of_action * joint_action_values[joint_action]

    # Compute the node value
    node_values[node] = policy_fn(node, agent_action_values, policy)

    return node_values[node]


def map_graph_by_depth(root):
    """ Traverse the graph. Return a mapping of node to horizon."""
    process_list = deque([root])
    depth_map = defaultdict(lambda: 0)
    depth_map[root] = 0

    # Queue all nodes in tree according to depth
    while process_list:
        node = process_list.pop()
        horizon = depth_map[node]

        for joint_action, successor_distribution in node.successors.items():
            for successor in successor_distribution:
                if horizon + 1 > depth_map[successor]:
                    # The successor can be deeper in certain branches.
                    depth_map[successor] = horizon + 1
                    process_list.append(successor)

    return depth_map


def create_node_set(node, node_set=None):
    if node_set is None:
        node_set = set()

    open_set = {node}
    while open_set:
        node = open_set.pop()
        node_set.add(node)
        open_set |= node.successor_set() - node_set

    return node_set


def detect_cycle(node):
    """
    Uses DFS to detect if there exist any cycles in the directed graph.
    """

    # Define recursive depth-first search function
    def dfs(current, unvisited, incomplete, complete):
        unvisited.remove(current)
        incomplete.add(current)

        for successor in current.successor_set():
            if successor in complete:
                continue
            elif successor in incomplete:
                return True
            elif dfs(successor, unvisited, incomplete, complete):
                return True

        incomplete.remove(current)
        complete.add(current)
        return False

    # Return result (all nodes are reachable from the node, so only one call is necessary)
    return dfs(node, create_node_set(node), set(), set())


def map_graph(node):
    """
    Builds a dict mapping of states to corresponding nodes in the graph.
    """
    return {n.state: n for n in create_node_set(node)}


def prune_unreachable(node, node_map, checked):
    """
    Prunes currently unreachable nodes from the graph, which cuts down on policy computation time for
    irrelevant areas of the state space.
    """
    checked.add(node)
    prune_set = set(pred for pred in node.predecessors if pred.state not in node_map)
    node.predecessors -= prune_set

    for pruned_node in prune_set:
        pruned_node.successors = {}

    for successor in [succ for succ in node.successor_set() if succ not in checked]:
        prune_unreachable(successor, node_map, checked)


def traverse_graph_topologically(depth_map, node_fn, top_down=True):
    """ Traverses the graph either top-down or bottom-up. """
    factor = 1 if top_down else -1
    node_list = list((factor * horizon, node) for node, horizon in depth_map.items())
    node_list.sort()

    for priority, node in node_list:
        node_fn(node, priority * factor)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    import sys
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

