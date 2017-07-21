from math import log
from heapq import heappush, heappop
from collections import defaultdict
from operator import mul
from functools import reduce

from agents.modeling_agent import get_max_action
from mdp.distribution import Distribution
from agents.communication.communication_scenario import Query, traverse_policy_graph, compute_reachable_nodes


def node_likelihoods(root):
    """
    Traverse graph. Return a mapping of node to probability
    """
    # TOP DOWN, doing each level together. Not depth first.
    process_list = [(0, root)]
    node_probabilities = defaultdict(float)
    node_probabilities[root] = 1.0

    # Queue all nodes in tree according to depth
    while process_list:
        level, node = heappop(process_list)
        node_probability = node_probabilities[node]

        if not node.successors:
            continue

        state = node.state
        joint_action_space = node.action_space

        # calculate prob of each joint action, giving agent's actions equal weight.
        model_predictions = {other_agent: other_agent_model.predict(state['World State'])
                             for other_agent, other_agent_model in state['Models'].items()}

        joint_action_probs = Distribution({joint_action: reduce(mul, [model_predictions[agent][joint_action[agent]]
                                                         for agent in model_predictions])
                                            for joint_action in joint_action_space})
        joint_action_probs.normalize()

        # pass probs onto successor nodes and add them to processing queue
        for joint_action, successor_distribution in node.successors.items():
            for successor, successor_probability in successor_distribution.items():
                # Add new layer of nodes to process. Don't duplicate in queue.
                if successor not in node_probabilities:
                    heappush(process_list, (level + 1, successor))

                # Total prob = parent prob * action prob * prob of resulting node from action
                node_probabilities[successor] += node_probability * joint_action_probs[joint_action] * successor_probability

    return node_probabilities


def get_active_node_set(root):
    """
    Traverse graph. Return set of non-terminal nodes with non-zero probability.
    """
    process_list = [root]
    added = {root}

    while process_list:
        node = process_list.pop()

        for successor in node.successor_set():
            if successor not in added:
                process_list.append(successor)
                added.add(successor)

    probs = node_likelihoods(root)

    return set(node for node in added if probs[node] > 0 and not node.scenario_end)


def entropy(root, eligible_states):
    """
    query = argmax(q) sum_{actions} P(action) * log( P(action) )
    """
    def entropy_calc(node, target_agent):
        # Update agent model
        agent_model = node.state['Models'][target_agent]

        # Predict action distribution
        predicted_actions = agent_model.predict(node.state['World State'])

        # Calculate entropy
        return sum(-1 * probability * log(probability) for probability in predicted_actions.values() if probability > 0)

    node_set = [node for node in get_active_node_set(root) if node.state['World State'] in eligible_states]
    possible_targets = root.state['Models'].keys()

    entropies = {}
    for target in possible_targets:
        for node in node_set:
            state = node.state['World State']
            query = Query(target, state)
            if state in entropies:
                existing = entropies[query]
                entropies[query] = max(existing, entropy_calc(node, target))
            else:
                entropies[query] = entropy_calc(node, target)

    return max(entropies, key=lambda q: entropies[q])


def weighted_entropy(root, eligible_states):
    """
    query = argmax(q) P(state) * [sum_{actions} P(action) * log( P(action) )]
    """
    def entropy_calc(node, target_agent):
        # Update agent model
        agent_model = node.state['Models'][target_agent]

        # Predict action distribution
        predicted_actions = agent_model.predict(node.state['World State'])

        # Calculate entropy
        return sum(-1 * probability * log(probability) for probability in predicted_actions.values() if probability > 0)

    node_set = [node for node in get_active_node_set(root) if node.state['World State'] in eligible_states]
    node_probs = node_likelihoods(root)
    possible_targets = root.state['Models'].keys()
    entropies = {}
    for target in possible_targets:
        for node in node_set:
            state = node.state['World State']
            query = Query(target, state)
            if state in entropies:
                existing = entropies[query]
                entropies[query] = max(existing, node_probs[node] * entropy_calc(node, target))
            else:
                entropies[query] = node_probs[node] * entropy_calc(node, target)

    return max(entropies, key=lambda q: entropies[q])


def variance_in_util(root, eligible_states):
    """
    query = argmax(q) sum_{actions} P(action) * (value - avg) * (value - avg)
    """
    def var_calc(node, target_agent):
        # Predict action distribution
        model_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                             for other_agent, other_agent_model in node.state['Models'].items()}

        joint_action_probs = Distribution({joint_action: reduce(mul, [model_predictions[agent][joint_action[agent]]
                                                         for agent in model_predictions])
                                            for joint_action in node.action_space})
        joint_action_probs.normalize()

        joint_action_values = node.action_values()
        agent_action_expectations = defaultdict(float)
        for joint_action, joint_action_prob in joint_action_probs.items():
            agent_action_expectations[joint_action[target_agent]] += joint_action_prob * joint_action_values[joint_action]

        return sum(action_prob * (agent_action_expectations[action] - node.future_value) ** 2
                   for action, action_prob in model_predictions[target_agent].items())

    node_set = [node for node in get_active_node_set(root) if node.state['World State'] in eligible_states]
    possible_targets = root.state['Models'].keys()
    entropies = {}
    for target in possible_targets:
        for node in node_set:
            state = node.state['World State']
            query = Query(target, state)
            if state in entropies:
                existing = entropies[query]
                entropies[query] = max(existing, var_calc(node, target))
            else:
                entropies[query] = var_calc(node, target)

    return max(entropies, key=lambda q: entropies[q])


def weighted_variance(root, eligible_states):
    """
    query = argmax(q) P(state) * [sum_{actions} P(action) * (value - avg) * (value - avg)]
    """
    def var_calc(node, target_agent):
        # Predict action distribution
        model_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                             for other_agent, other_agent_model in node.state['Models'].items()}

        joint_action_probs = Distribution({joint_action: reduce(mul, [model_predictions[agent][joint_action[agent]]
                                                                      for agent in model_predictions])
                                           for joint_action in node.action_space})
        joint_action_probs.normalize()

        joint_action_values = node.action_values()
        agent_action_expectations = defaultdict(float)
        for joint_action, joint_action_prob in joint_action_probs.items():
            agent_action_expectations[joint_action[target_agent]] += joint_action_prob * joint_action_values[
                joint_action]

        return sum(action_prob * (agent_action_expectations[action] - node.future_value) ** 2
                   for action, action_prob in model_predictions[target_agent].items())

    node_set = [node for node in get_active_node_set(root) if node.state['World State'] in eligible_states]
    node_probs = node_likelihoods(root)
    possible_targets = root.state['Models'].keys()
    entropies = {}
    for target in possible_targets:
        for node in node_set:
            state = node.state['World State']
            query = Query(target, state)
            if state in entropies:
                existing = entropies[query]
                entropies[query] = max(existing, node_probs[node] * var_calc(node, target))
            else:
                entropies[query] = node_probs[node] * var_calc(node, target)

    return max(entropies, key=lambda q: entropies[q])


def myopic(root, agent_identity, eligible_states):
    """
    query = argmax(q) E[V(s_0) | \pi']
    """
    def calc_ev(agent, model, query, action):
        def compute_policy(node, action_values, new_policy):
            action, action_value = max(action_values.items(), key=lambda pair: pair[1])
            new_policy[node.state] = action
            return action_value

        new_model = model.communicated_policy_update([(query, action)])
        new_model_state = root.state['Models'].update({agent: new_model})
        policy = {}
        expected_util = traverse_policy_graph(node=root, node_values={}, model_state=new_model_state,
                                              policy=policy, policy_fn=compute_policy,
                                              agent_identity=agent_identity)

        return expected_util

    # For every state, we need to consider what the new policy would be.
    # Should be able to use functions from comm scenario.
    query_ev = {}
    for node in [node for node in get_active_node_set(root) if node.state['World State'] in eligible_states]:
        world_state = node.state['World State']
        model_state = node.state['Models']
        for agent, model in model_state.items():
            query = Query(agent, world_state)
            query_ev[query] = sum(prob*calc_ev(agent, model, query, action)
                                                       for action, prob in model.predict(world_state).items())

    return max(query_ev, key=lambda q: query_ev[q])



