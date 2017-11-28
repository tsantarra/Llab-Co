from math import log
from collections import defaultdict
from operator import mul
from functools import reduce

from mdp.distribution import Distribution
from agents.communication.communication_scenario import Query
from mdp.graph_planner import create_node_set


def example_heuristic(policy_root, target_agent_model, prune_fn):
    # Calculate aggregate info (graph traversals, etc)

    # Create set of candidate nodes, calling the prune_fn to initially cut down the set

    # Evaluate each remaining node/state. Keep state and value in a non-destructive manner,
    # as different nodes can refer to the same state (resulting in different evaluations; should we
    # combine them? No. Then, the most common state may trump more informative state. Right?)

    # Return query evaluations.
    pass


def entropy_calc(policy_root, target_agent, prune_fn):
    """
    query = argmax(q) sum_{actions} P(action) * log( P(action) )
    """
    def evaluate(node, target_agent_model):
        # Predict action distribution
        predicted_actions = target_agent_model.predict(node.state['World State'])

        # Calculate entropy
        return sum(-1 * probability * log(probability) for probability in predicted_actions.values() if probability > 0)

    # Calculate aggregate info (graph traversals, etc)
    #node_probs = node_likelihoods(policy_root)
    candidate_nodes = set()
    create_node_set(policy_root, candidate_nodes)

    # Create set of candidate nodes, calling the prune_fn to initially cut down the set
    candidate_nodes = set(node for node in candidate_nodes if not prune_fn(node))

    # Evaluate each remaining node.
    node_evals = [(node, target_agent_model.name, evaluate(nod))]
    #need updated model, not the one from the policy graph
    # need state from QUERY mdp (so as to include all updated ev)
    # which means we should store EV of every node as well...

    # Return query evaluations.



def weighted_entropy(node, target_agent_model, node_probability):
    """
    query = argmax(q) P(state) * [sum_{actions} P(action) * log( P(action) )]
    """
    return node_probability * entropy_calc(node, target_agent_model, node_probability)


def variance_in_util(node, target_agent_model, node_probability):
    """
    query = argmax(q) sum_{actions} P(action) * (value - avg) * (value - avg)
    """
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


def weighted_variance(node, target_agent, node_probability):
    """
    query = argmax(q) P(state) * [sum_{actions} P(action) * (value - avg) * (value - avg)]
    """
    return node_probability * variance_in_util(node, target_agent, node_probability)


def create_myopic_heuristic(root, agent_identity):
    def myopic(node, target_agent, node_probability):
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

        # Info needed to calculate new policy's expected value.
        world_state = node.state['World State']
        model = node.state['Models'][target_agent]
        query = Query(target_agent, world_state)

        # Return the expectation over responses.
        return sum(prob * calc_ev(target_agent, model, query, action)
                   for action, prob in model.predict(world_state).items())

    return myopic


