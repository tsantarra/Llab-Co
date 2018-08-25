from math import log
from collections import defaultdict
from random import randint
from operator import mul
from functools import reduce

from agents.communication.communication_scenario import Query
from mdp.graph_utilities import recursive_traverse_policy_graph, traverse_graph_topologically, map_graph_by_depth
from agents.modeling_agent import individual_agent_action_values


"""
        
        V(s_0) = sum_{s} λ^t E_{s,a,s'}[R(s,a,s')]
        V(s_0) = sum_{s} λ^t p(a_j |s) p(s) T(s,a,s') R(s,a,s')                                 # T(s,a,s') = P(s'|s,a)
        ΔV(s_0) = sum_{s_t} λ^t T(s,a,s') R(s,a,s') [ p'(a_j|s) p'(s)  - p(a_j|s) p(s) ]
        
        Interpretation: 
            ΔV is related to the change in policy prediction (p(a|s)) and trajectory (p(s)), as weighted by the 
            discounted expected immediate reward. 
        Observations:
            p(a|s) is dependent on V(s') and, as such, all of the subgraph for t = [t+1, t+2, ... ].
            p(s) is dependent on all of the policy decisions (p(a|s)) in t = [0, t-1]
        Conclusion:
            In order for a local estimate to be made, either or both of these components must have estimated changes,
            possibly fixing the value (assuming it won't change with new knowledge). For heuristics that don't
            estimate this change (but rather estimate correlated information), globally fixed metrics are likely
            required as well. 
            
        Assuming independence from all other states and state decisions -> greedy local optimizations.
            "nonetheless a greedy heuristic may yield locally optimal solutions that approximate a global optimal 
             solution in a reasonable time."
             
            Plus, we're just using a greedy heuristic to order the queries at a given information state. As you 
            widen the search, you are guaranteed to converge to the correct optimal query once it is included. 
            
            Should plot heuristic value (x) and query value (y) to show correlations between heuristic ordering and
            actual query policy value. If doing across trials with diff constraints, need to normalize value. 

        Heuristics:
            Information Entropy             sum_a p(a) [ log p(a) ]
            Value of Information            sum_a p(a) [ V(s | a, π') - V(s | a, π)]
            Absolute Error                  sum_a p(a) [ | V(s | a) - V(s) | ]
            Utility Variance                sum_a p(a) [ V(s | a) - V(s)]^2
            
            Conditional Value at Risk       P( V(s|a,π) < V(s|π) )  # likelihood of current overestimation
            Conditional Value to Gain       P( V(s|a,π) > V(s|π) )  # likelihood of current underestimation
                Note: Target values for this CDF are dependent on earlier decisions (e.g. how much do I need to 
                      have underestimated at s_t to influence a decision at s_{t-1} or even s_0?). 
                      We can, however, give a local target value of V(s|π'), where π' is the second best policy. 
            
            myopic                          sum_{p(a_t)} [ V(s_0 | a_t, π') - V(s_0 | a_t, π)]
                                                        
            random                          rand()
            most likely next state          p(s)
            

"""


def example_heuristic(policy_root, target_agent_name, prune_fn):
    """
    Args:
        policy_root:            A pointer to the root node for the policy graph paired with the target_agent_model
        target_agent_name:      The name of the agent we're considering querying.
        prune_fn:               A function that determines if a node would be ruled out from communication.

        *Note: INCORRECT** The target agent model is the model of the agent at the root of the policy graph, i.e. is the
               set of probabilities associates with the agent's policy at the given time, not at the project state time,
               which may have refined its policy predictions by observing behavior along the state trajectory between
               now and then. As we have not yet observed that trajectory, we cannot be certain that an accurate
               information state. All we can be sure of is (1) what we have observed until now and (2) what we have
               communicated thus far.

        **Note: I was mistaken. We need only consider the uncertainty at s_t. The local heuristics are meant to be
                projections of uncertainty/value at the time, as it doesn't make sense to query based on current
                uncertainty if we may infer certainty by an intermediate observation we just haven't gotten to (but will
                occur).

    Returns:
        A list of query, heuristic value pairs to be used in choosing policy query actions. (higher value => better)
    """
    # Calculate aggregate info (graph traversals, etc)

    # Create set of candidate nodes, calling the prune_fn to initially cut down the set

    # Evaluate each remaining node/state. Note that states are not necessarily unique (trajectories are unique), but
    # nodes containing matching states may have varying local heuristic values. As we are interested in keeping
    # only a set of n states with maximal heuristic values, simply keep the maximal value encountered per unique
    # state. 

    # Return query evaluations.
    pass


def local_information_entropy(policy_root, target_agent_name, prune_fn, agent_name, gamma=1.0):
    """
    Weight by lambda^t until we can figure out a better estimate of the probabilities involved in:
        Information Entropy           λ^t sum_{p(a)} [ p(a) log p(a) ]
    """
    eval_list = []

    def evaluate(node, horizon):
        if prune_fn(node, target_agent_name):
            return

        # Calculate entropy
        predicted_actions = node.state['Models'][target_agent_name].predict(node.state['World State'])
        eval_list.append((node.state['World State'],
                         (gamma ** horizon) * sum(-1 * probability * log(probability)
                                                  for probability in predicted_actions.values() if probability > 0)))

    depth_map = map_graph_by_depth(policy_root)
    i = len(depth_map)
    traverse_graph_topologically(depth_map, evaluate)
    return eval_list


def local_value_of_information(policy_root, target_agent_name, prune_fn, agent_name, gamma=1.0):
    """
    Value of Information            sum_{p(a)} [ V(s | a, π') - V(s | a, π)]
    """
    eval_list = []

    def evaluate(node, horizon):
        if prune_fn(node, target_agent_name):
            return

        # Calculate entropy
        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}

        # needs identity
        old_action_values = individual_agent_action_values(agent_name, other_agent_predictions, node.action_space,
                                                           node.action_values())
        old_policy_action, old_policy_old_val = max(old_action_values.items(), key=lambda pair: pair[1])

        calc_value = 0
        for teammate_action, probability in other_agent_predictions[target_agent_name].items():
            # Q(s,a) with model updated and policy changed
            other_agent_predictions[target_agent_name] = defaultdict(float)
            other_agent_predictions[target_agent_name][teammate_action] = 1.0

            new_action_values = individual_agent_action_values(agent_name, other_agent_predictions, node.action_space,
                                                               node.action_values())

            # sum_{responses} P(response) [V(new policy action, new knowledge) - V(old policy action, new knowledge)]
            calc_value += probability * (max(new_action_values.values()) - new_action_values[old_policy_action])

        eval_list.append((node.state['World State'], (gamma ** horizon) * calc_value))

    traverse_graph_topologically(map_graph_by_depth(policy_root), evaluate)
    return eval_list


def local_absolute_error(policy_root, target_agent_name, prune_fn, agent_name, gamma=1.0):
    """
    Absolute Error                  sum_{p(a)} [ V(s | a) - V(s)]
    """
    eval_list = []

    def evaluate(node, horizon):
        if prune_fn(node, target_agent_name):
            return

        # Calculate entropy
        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}

        # needs identity
        old_action_values = individual_agent_action_values(agent_name, other_agent_predictions, node.action_space,
                                                           node.action_values())
        policy_action, policy_old_value = max(old_action_values.items(), key=lambda pair: pair[1])

        # We need the expected value conditioned on the teammate's action, so we use individual_agent_action_values,
        # but we use our policy as a prediction and leave open the teammate's policy (inverse of normal usage).
        other_agent_predictions[agent_name] = defaultdict(float)
        other_agent_predictions[agent_name][policy_action] = 1.0
        teammate_predictions = other_agent_predictions[target_agent_name]
        del other_agent_predictions[target_agent_name]

        teammate_action_values = individual_agent_action_values(target_agent_name,
                                                                other_agent_predictions,
                                                                node.action_space,
                                                                node.action_values())

        expected_absolute_error = sum(prob * abs(teammate_action_values[action] - node.future_value)
                                      for action, prob in teammate_predictions.items)

        eval_list.append((node.state['World State'], (gamma ** horizon) * expected_absolute_error))

    traverse_graph_topologically(map_graph_by_depth(policy_root), evaluate)
    return eval_list


def local_utility_variance(policy_root, target_agent_name, prune_fn, agent_name, gamma=1.0):
    """
    Utility Variance                sum_{p(a)} [ V(s | a) - V(s)]^2
    """
    eval_list = []

    def evaluate(node, horizon):
        if prune_fn(node, target_agent_name):
            return

        # Calculate entropy
        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}

        # needs identity
        old_action_values = individual_agent_action_values(agent_name,
                                                           other_agent_predictions,
                                                           node.action_space,
                                                           node.action_values())
        policy_action, policy_old_value = max(old_action_values.items(), key=lambda pair: pair[1])

        # We need the expected value conditioned on the teammate's action, so we use individual_agent_action_values,
        # but we use our policy as a prediction and leave open the teammate's policy (inverse of normal usage).
        other_agent_predictions[agent_name] = defaultdict(float)
        other_agent_predictions[agent_name][policy_action] = 1.0
        teammate_predictions = other_agent_predictions[target_agent_name]
        del other_agent_predictions[target_agent_name]

        teammate_action_values = individual_agent_action_values(target_agent_name,
                                                                other_agent_predictions,
                                                                node.action_space,
                                                                node.action_values())

        expected_absolute_error = sum(prob * pow(teammate_action_values[action] - node.future_value, 2)
                                      for action, prob in teammate_predictions.items)

        eval_list.append((node.state['World State'], (gamma ** horizon) * expected_absolute_error))

    traverse_graph_topologically(map_graph_by_depth(policy_root), evaluate)
    return eval_list


def random_evaluation(policy_root, target_agent_name, prune_fn, agent_name):
    """
    Give random evaluations, as a comparison baseline.
    """
    return [(node.state['World State'], randint()) for node in map_graph_by_depth(policy_root)]


def most_likely_next_state(policy_root, target_agent_name, prune_fn, agent_name):
    """
    Order states by most likely - p(s | π)
    """
    node_probs = defaultdict(float)
    node_probs[policy_root] = 1.0

    def evaluate(node, horizon):
        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}
        old_action_values = individual_agent_action_values(agent_name, other_agent_predictions, node.action_space,
                                                           node.action_values())
        policy_action, policy_old_value = max(old_action_values.items(), key=lambda pair: pair[1])

        for joint_action in node.action_space.fix_actions({agent_name: [policy_action]}):
            for successor, successor_prob in node.successors[joint_action].items():
                node_probs[successor] += node_probs[node] * successor_prob * \
                                         reduce(mul, [other_agent_predictions[other_agent][joint_action[other_agent]]
                                                      for other_agent in other_agent_predictions])

    traverse_graph_topologically(map_graph_by_depth(policy_root), evaluate)
    return [(node.state['World State'], prob) for node, prob in node_probs.items()]


def create_myopic_heuristic(root, agent_identity):
    """
    query = argmax(q=s_t) sum_{p(a_t)} [ V(s_0 | a_t, π') - V(s_0 | a_t, π)]
    """
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
            expected_util = recursive_traverse_policy_graph(node=root, node_values={}, model_state=new_model_state,
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


