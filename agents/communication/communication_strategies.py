"""     Notes on Heuristic Query Evaluators

        Classifications of Heuristics:
            Immediate vs Future
                Immediate               - affecting the current decision/uncertainty
                Future                  - affecting the decision/uncertainty at the state of the query
            Information-theoretic vs Decision-theoretic
                Information-theoretic   - concerned solely with uncertainty
                Decision-theoretic      - a mixture of uncertainty and utility
            Fixed vs Potential-based
                Fixed                   - based on the characteristics around the state of the query
                Potential-based         - a change in characteristic given a change in information

        Heuristics with classifications (at I_0 = {s_0, M_0} and evaluating query at I_t = {s_t, M_t}):
        (W) Information entropy over actions  (Δ would be equivalent, given drop to 0)
                sum_{a_i} p(a_i | M_t) log p(a_i | M_t)   (at s_t)
                future, fixed, information-theoretic
        (W) Expected absolute loss
                E_π[ |V(s_t | a_i) - V(s)| ]
                future, fixed, decision-theoretic
        (W) Expected mean squared error
                E_π[ (V(s_t | a_i) - V(s))^2 ]
                future, fixed, decision-theoretic

        (W) Δ Future information entropy over policies
                E_{π_i(s_t)}[ sum_{π_i} p'(π_i | M_t) log p'(π_i | M_t) ] - sum_{π_i} p(π_i | M_t) log p(π_i | M_t)
                future, potential-based, information-theoretic
        (W) Future value of information
                E_{M_t}[ V(s_t | π') - V(s_t | π) ]
                future, potential-based, decision-theoretic

            Δ Immediate information entropy over policies
                E_{π_i(s_t)}[ Δ sum_{π_i} p(π_i | M_0) log p(π_i | M_0) ]
                future, potential-based, information-theoretic
            Immediate value of information
                E_{M_0}[ V(s_0 | π_i(s_t)) - V(s_0 | π_i(s_t)) ]
                future, potential-based, decision-theoretic

        (W) = weighted


        How value plays a role:
            V(s_0) = sum_{s} λ^t E_{s,a,s'}[R(s,a,s')]
            V(s_0) = sum_{s} λ^t p(a_j |s) p(s) T(s,a,s') R(s,a,s')                          # T(s,a,s') = P(s'|s,a)
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

        Heuristic ideas:
            Information Entropy             sum_a p(a) [ log p(a) ]
            Value of Information            sum_a p(a) [ V(s | a, π') - V(s | a, π)]
            Absolute Loss                  sum_a p(a) [ | V(s | a) - V(s) | ]
            Variance/Quadratic Loss         sum_a p(a) [ V(s | a) - V(s)]^2
            
            Conditional Value at Risk       P( V(s|a,π) < V(s|π) )  # likelihood of current overestimation
            Conditional Value to Gain       P( V(s|a,π) > V(s|π) )  # likelihood of current underestimation
                Note: Target values for this CDF are dependent on earlier decisions (e.g. how much do I need to 
                      have underestimated at s_t to influence a decision at s_{t-1} or even s_0?). 
                      We can, however, give a local target value of V(s|π'), where π' is the second best policy. 
            
            myopic                          sum_{p(a_t)} [ V(s_0 | a_t, π') - V(s_0 | a_t, π)]
            random                          rand()
            most likely next state          p(s)
"""
from math import log, exp
from collections import defaultdict
from random import random
from operator import mul
from functools import reduce

from agents.communication.communication_scenario import Query
from agents.modeling_agent import individual_agent_action_values
from mdp.graph_utilities import recursive_traverse_policy_graph, traverse_graph_topologically
from mdp.distribution import ListDistribution


def example_heuristic(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
    Args:
        policy_root:            A pointer to the root node for the policy graph paired with the target_agent_model
        depth_map:              A horizon-based map of the policy graph
        target_agent_name:      The name of the agent we're considering querying.
        agent_identity:         The name of the agent communicating.
        prune_fn:               A function that determines if a node would be ruled out from communication.
        gamma:                  The discount factor used in the scenario.

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


def weighted(heuristic):
    def weighted_heuristic(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
        # Calculate weights
        probs = [1.0]
        node_probs = defaultdict(float)
        node_probs[policy_root] = 1.0

        def calculate_node_likelihoods(node, _):

            prob = node_probs[node]
            if not prune_fn(node, target_agent_name):
                probs.append(prob)

            if not node.action_space:  # terminal node
                return

            other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                       for other_agent, other_agent_model in node.state['Models'].items()}
            action_values = individual_agent_action_values(agent_identity, other_agent_predictions, node.action_space,
                                                               node.action_values())
            action_distribution = ListDistribution([(action, exp(value)) for action, value in action_values.items()])\
                                    .normalize()

            for action, action_prob in action_distribution.items():
                for joint_action in node.action_space.fix_actions({agent_identity: [action]}):
                    for successor, successor_prob in node.successors[joint_action].items():
                        node_probs[successor] += action_prob * successor_prob * \
                                                 reduce(mul,
                                                        [other_agent_predictions[other_agent][joint_action[other_agent]]
                                                         for other_agent in other_agent_predictions])

        traverse_graph_topologically(depth_map, calculate_node_likelihoods, top_down=True)

        results = heuristic(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma)

        return [(state, val * prob) for (state, val), prob in zip(results, probs)]

    return weighted_heuristic


def local_action_information_entropy(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
        (W) Information entropy over actions  (Δ would be equivalent, given drop to 0)
            sum_{a_i} p(a_i | M_t) log p(a_i | M_t)   (at s_t)
            future, fixed, information-theoretic
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

    traverse_graph_topologically(depth_map, evaluate)
    return eval_list


def local_absolute_error(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
        (W) Expected absolute loss
                E_π[ |V(s_t | a_i) - V(s)| ]
                future, fixed, decision-theoretic
    """
    eval_list = []

    def evaluate(node, horizon):
        if prune_fn(node, target_agent_name):
            return

        # Calculate entropy
        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}

        # needs identity
        old_action_values = individual_agent_action_values(agent_identity, other_agent_predictions, node.action_space,
                                                           node.action_values())
        policy_action, policy_old_value = max(old_action_values.items(), key=lambda pair: pair[1])

        # We need the expected value conditioned on the teammate's action, so we use individual_agent_action_values,
        # but we use our policy as a prediction and leave open the teammate's policy (inverse of normal usage).
        other_agent_predictions[agent_identity] = defaultdict(float)
        other_agent_predictions[agent_identity][policy_action] = 1.0
        teammate_predictions = other_agent_predictions[target_agent_name]
        del other_agent_predictions[target_agent_name]

        teammate_action_values = individual_agent_action_values(target_agent_name,
                                                                other_agent_predictions,
                                                                node.action_space,
                                                                node.action_values())

        expected_absolute_error = sum(prob * abs(teammate_action_values[action] - node.future_value)
                                      for action, prob in teammate_predictions.items())

        eval_list.append((node.state['World State'], (gamma ** horizon) * expected_absolute_error))

    traverse_graph_topologically(depth_map, evaluate)
    return eval_list


def local_mean_squared_error(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
        (W) Expected mean squared error
                E_π[ (V(s_t | a_i) - V(s))^2 ]
                future, fixed, decision-theoretic
    """
    eval_list = []

    def evaluate(node, horizon):
        if prune_fn(node, target_agent_name):
            return

        # Calculate entropy
        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}

        # needs identity
        old_action_values = individual_agent_action_values(agent_identity,
                                                           other_agent_predictions,
                                                           node.action_space,
                                                           node.action_values())
        policy_action, policy_old_value = max(old_action_values.items(), key=lambda pair: pair[1])

        # We need the expected value conditioned on the teammate's action, so we use individual_agent_action_values,
        # but we use our policy as a prediction and leave open the teammate's policy (inverse of normal usage).
        other_agent_predictions[agent_identity] = defaultdict(float)
        other_agent_predictions[agent_identity][policy_action] = 1.0
        teammate_predictions = other_agent_predictions[target_agent_name]
        del other_agent_predictions[target_agent_name]

        teammate_action_values = individual_agent_action_values(target_agent_name,
                                                                other_agent_predictions,
                                                                node.action_space,
                                                                node.action_values())

        mean_squared_error = sum(prob * pow(teammate_action_values[action] - node.future_value, 2)
                                      for action, prob in teammate_predictions.items())

        eval_list.append((node.state['World State'], (gamma ** horizon) * mean_squared_error))

    traverse_graph_topologically(depth_map, evaluate)
    return eval_list


def local_delta_policy_entropy(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
        (W) Δ Future information entropy over policies
                E_{π_i(s_t)}[ sum_{π_i} p'(π_i | M_t) log p'(π_i | M_t) ] - sum_{π_i} p(π_i | M_t) log p(π_i | M_t)
                future, potential-based, information-theoretic
    """
    eval_list = []

    def evaluate(node, _):
        if prune_fn(node, target_agent_name):
            return

        # Get model
        state = node.state['World State']
        communicating_teammate_model = node.state['Models'][target_agent_name]

        # -1 * E[ΔH] = H - E[H | a]  -> we flip it because we end at lower entropy, but we're MAXing values
        expected_entropy_diff = sum(-1 * prob * log(prob) for policy_index, prob
                                        in communicating_teammate_model.model.policy_distribution if prob > 0)

        expected_entropy_diff -= sum(probability * sum(-1 * policy_prob * log(policy_prob)
                                                       for policy_index, policy_prob
                                                       in communicating_teammate_model.update(state, prediction).model.policy_distribution
                                                       if policy_prob > 0)
                                     for prediction, probability in communicating_teammate_model.predict(state).items())

        eval_list.append((state, expected_entropy_diff))

    traverse_graph_topologically(depth_map, evaluate)
    return eval_list


def local_value_of_information(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
        (W) Future value of information
                E_{M_t}[ V(s_t | π') - V(s_t | π) ]
                future, potential-based, decision-theoretic
    """
    eval_list = []

    def evaluate(node, horizon):
        if prune_fn(node, target_agent_name):
            return

        # Calculate old policy (to be evaluated under new action values)
        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}

        old_action_values = individual_agent_action_values(agent_identity, other_agent_predictions, node.action_space,
                                                           node.action_values())
        old_policy_action, old_policy_old_val = max(old_action_values.items(), key=lambda pair: pair[1])

        value_of_info = 0
        for teammate_action, probability in other_agent_predictions[target_agent_name].items():
            # Q(s,a) with model updated and policy changed
            other_agent_predictions[target_agent_name] = defaultdict(float)
            other_agent_predictions[target_agent_name][teammate_action] = 1.0

            new_action_values = individual_agent_action_values(agent_identity, other_agent_predictions, node.action_space,
                                                               node.action_values())

            # sum_{responses} P(response) [V(new policy action, new knowledge) - V(old policy action, new knowledge)]
            value_of_info += probability * (max(new_action_values.values()) - new_action_values[old_policy_action])

        eval_list.append((node.state['World State'], (gamma ** horizon) * value_of_info))

    traverse_graph_topologically(depth_map, evaluate)
    return eval_list


def immediate_delta_policy_entropy(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
        Δ Immediate information entropy over policies
            E_{π_i(s_t)}[ Δ sum_{π_i} p(π_i | M_0) log p(π_i | M_0) ]
            future, potential-based, information-theoretic
    """
    eval_list = []

    root_teammate_model = policy_root.state['Models'][target_agent_name]
    base_entropy = sum(-1 * prob * log(prob) for policy_index, prob in root_teammate_model.model.policy_distribution
                       if prob > 0)

    def evaluate(node, _):
        if prune_fn(node, target_agent_name):
            return

        # E[ΔH] = E[H | a] - H
        future_state = node.state['World State']
        expected_root_entropy = sum(probability *
                                    sum(-1 * policy_prob * log(policy_prob)
                                        for policy_index, policy_prob
                                        in root_teammate_model.update(future_state, prediction).model.policy_distribution
                                        if policy_prob > 0)
                                    for prediction, probability in root_teammate_model.predict(future_state).items())

        eval_list.append((future_state, base_entropy - expected_root_entropy))

    traverse_graph_topologically(depth_map, evaluate)
    return eval_list


def immediate_approx_value_of_information(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
        Immediate value of information
            E_{M_0}[ V(s_0 | π_i(s_t)) - V(s_0 | π_i(s_t)) ]
            future, potential-based, decision-theoretic
    """
    eval_list = []

    root_state = policy_root.state['World State']
    root_teammate_model = policy_root.state['Models'][target_agent_name]
    teammate_predictions = {other_agent: other_agent_model.predict(root_state)
                               for other_agent, other_agent_model in policy_root.state['Models'].items()}

    old_action_values = individual_agent_action_values(agent_identity, teammate_predictions, policy_root.action_space,
                                                       policy_root.action_values())
    old_policy_action, _ = max(old_action_values.items(), key=lambda pair: pair[1])

    def evaluate(node, _):
        if prune_fn(node, target_agent_name):
            return

        value_of_info = 0
        future_state = node.state['World State']

        # Expectation of root model predictions of FUTURE state policy (query/response)
        # Value of knowing (query/response) NOW (root)
        for future_action, future_action_prob in root_teammate_model.predict(future_state).items():
            # Q(s,a) with model updated and policy changed
            new_model = root_teammate_model.update(future_state, future_action)
            new_root_prediction = new_model.predict(root_state)

            new_agent_predictions = {agent: predictions if agent != target_agent_name else new_root_prediction
                                     for agent, predictions in teammate_predictions.items()}

            new_action_values = individual_agent_action_values(agent_identity, new_agent_predictions,
                                                               policy_root.action_space,
                                                               policy_root.action_values())

            # sum_{responses} P(response) [V(new policy action, new knowledge) - V(old policy action, new knowledge)]
            value_of_info += future_action_prob * (max(new_action_values.values()) - new_action_values[old_policy_action])

        eval_list.append((node.state['World State'], value_of_info))

    traverse_graph_topologically(depth_map, evaluate)
    return eval_list



def random_evaluation(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
    Give random evaluations, as a comparison baseline.
    """
    return [(node.state['World State'], random()) for node in depth_map if not prune_fn(node, target_agent_name)]


def state_likelihood(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
    """
    Order states by most likely - p(s | π)
    """
    node_probs = defaultdict(float)
    node_probs[policy_root] = 1.0

    def evaluate(node, _):
        if prune_fn(node, target_agent_name):
            return

        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}
        old_action_values = individual_agent_action_values(agent_identity, other_agent_predictions, node.action_space,
                                                           node.action_values())
        policy_action, policy_old_value = max(old_action_values.items(), key=lambda pair: pair[1])

        for joint_action in node.action_space.fix_actions({agent_identity: [policy_action]}):
            for successor, successor_prob in node.successors[joint_action].items():
                node_probs[successor] += node_probs[node] * successor_prob * \
                                         reduce(mul, [other_agent_predictions[other_agent][joint_action[other_agent]]
                                                      for other_agent in other_agent_predictions])

    traverse_graph_topologically(depth_map, evaluate)
    return [(node.state['World State'], prob) for node, prob in node_probs.items()]


def create_myopic_heuristic(policy_root, depth_map, target_agent_name, agent_identity, prune_fn, gamma=1.0):
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
            new_model_state = policy_root.state['Models'].update_item(agent, new_model)
            policy = {}
            #### Update this
            expected_util = recursive_traverse_policy_graph(node=policy_root, node_values={},
                                                            model_state=new_model_state,
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

    raise Exception('Not checked for correctness. Verify before using.')

    return myopic


