from collections import defaultdict, deque
from operator import mul
from functools import reduce

from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace


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


def node_likelihoods(root):
    """
    Traverse the graph. Return a mapping of node to probability.
    """
    # Traverse graph. Pop off process list. Process. Decrement pred count for child. If zero, add to process set.
    process_list = [root]
    preds_left_to_process = dict()
    node_probabilities = defaultdict(float)
    node_probabilities[root] = 1.0

    # Queue all nodes in tree according to depth
    while process_list:
        node = process_list.pop()
        node_probability = node_probabilities[node]

        if not node.successors:
            continue

        # Calculate prob of each joint action, giving agent's actions equal weight.
        model_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                             for other_agent, other_agent_model in node.state['Models'].items()}

        joint_action_space = node.action_space
        joint_action_probs = Distribution({joint_action: reduce(mul, [model_predictions[agent][joint_action[agent]]
                                                                      for agent in model_predictions])
                                           for joint_action in joint_action_space})
        joint_action_probs.normalize()

        # pass probs onto successor nodes and add them to processing queue
        for joint_action, successor_distribution in node.successors.items():
            for successor, successor_probability in successor_distribution.items():
                # Add action-specific chance of arriving at the child node
                node_probabilities[successor] += \
                    node_probability * joint_action_probs[joint_action] * successor_probability

                # Add new layer of nodes to process only if all predecessors have been processed.
                if node not in preds_left_to_process:
                    preds_left_to_process[node] = len(node.predecessors)

                preds_left_to_process[node] -= 1
                if preds_left_to_process[node] == 0:
                    process_list.append(successor)

    return node_probabilities


def get_active_node_set(root):
    """
    Traverse graph. Return set of non-terminal nodes with non-zero probability.
    """
    probs = node_likelihoods(root)

    return set(node for node, prob in probs.items() if prob > 0 and not node.scenario_end)


def compute_reachable_nodes(node, visited_set, model_state):
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    visited_set.add(node)

    if not node.successors:
        return

    world_state = node.state['World State']
    predicted_actions = {other_agent: set(action for action, prob in other_agent_model.predict(world_state).items()
                                          if prob > 0)
                         for other_agent, other_agent_model in model_state.items()}

    resulting_models = {agent_name: {action: model_state[agent_name].update(world_state, action) for action in
                                     agent_actions}
                        for agent_name, agent_actions in predicted_actions.items()
                        if agent_name in model_state}

    individual_actions = node.action_space.individual_actions().copy()
    individual_actions.update(predicted_actions)

    for joint_action in JointActionSpace(individual_actions):
        # Update model state
        new_model_state = State({agent_name: resulting_models[agent_name][joint_action[agent_name]]
                                 for agent_name in model_state})

        # Traverse through applicable successor nodes
        for successor_node in (successor for successor in node.successors[joint_action] if
                               successor not in visited_set):
            compute_reachable_nodes(successor_node, visited_set, new_model_state)


def min_exp_util_fixed_policy(node, node_values, agent_identity, policy, policy_commitments):
    """ Searching for minimum expected utility within the given policy. """
    # Already covered this node and subgraph
    if node in node_values:
        return

    # Leaf node. Simply the node's future value.
    if not node.successors:
        node_values[node] = node.future_value
        return

    # Calculate new minimum expected util over action space.
    action_space = node.action_space

    # Set commitments
    if node.state in policy_commitments:
        action_space = action_space.constrain(
            dict(**policy_commitments[node.state], **{agent_identity: policy[node.state]}))
    else:
        action_space = action_space.constrain({agent_identity: [policy[node.state]]})

    # Recurse through only relevant portions of the subgraph.
    for successor in set(successor for action in action_space for successor in node.successors[action]):
        min_exp_util_fixed_policy(successor, node_values, agent_identity, policy, policy_commitments)

    # Construct new joint action space given constraints. Calculate new expected utils for each joint action.
    action_values = {
        action: sum(probability * (node.successor_transition_values[(successor.state, action)] + node_values[successor])
                    for successor, probability in node.successors[action].items())
        for action in action_space}

    node_values[node] = min(action_values.values())


def max_exp_util_free_policy(node, node_values, policy_commitments):
    """ Search for maximum expected utility given one or more policy changes. """
    # Already covered this node and subgraph
    if node in node_values:
        return

    # Leaf node. Simply the node's future value.
    if not node.successors:
        node_values[node] = node.future_value
        return

    # Calculate max minimum expected util. Only consider actions that match previous commitments!
    action_space = node.action_space

    # Set commitments
    if node.state in policy_commitments:
        action_space = action_space.constrain(policy_commitments[node.state])

    # Recurse through subgraph.
    for successor in set(successor for action in action_space for successor in node.successors[action]):
        max_exp_util_free_policy(successor, node_values, policy_commitments)

    # Construct new joint action space given constraints. Calculate new expected utils for each joint action.
    action_values = {action: sum(probability * (node.successor_transition_values[(successor.state, action)] +
                                                node_values[successor])
                                 for successor, probability in node.successors[action].items())
                     for action in action_space}

    node_values[node] = max(action_values.values())
