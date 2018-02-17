from math import log
from collections import defaultdict
from operator import mul
from functools import reduce, partial

from mdp.distribution import Distribution
from mdp.graph_planner import greedy_action
from agents.communication.communication_scenario import Query
from agents.communication.graph_utilities import recursive_traverse_policy_graph, traverse_graph_topologically, map_graph_by_depth
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


def example_heuristic(policy_root, agent_name, prune_fn):
    """
    Args:
        policy_root:            A pointer to the root node for the policy graph paired with the target_agent_model
        target_agent_model:     The model against which the communicating agent would be querying*.
        prune_fn:               A function that determines if a node would be ruled out from communication.

        *Note: The target agent model is the model of the agent at the root of the policy graph, i.e. is the set of
               probabilities associates with the agent's policy at the given time, not at the project state time, which
               may have refined its policy predictions by observing behavior along the state trajectory between now and
               then. As we have not yet observed that trajectory, we cannot be certain that an accurate information
               state. All we can be sure of is (1) what we have observed until now and (2) what we have communicated
               thus far.

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


def local_information_entropy(policy_root, target_agent, prune_fn):
    """
    Information Entropy             sum_{p(a)} [ p(a) log p(a) ]
    """
    eval_list = []

    def evaluate(node):
        """

        """
        if prune_fn(node):
            return

        # Calculate entropy
        predicted_actions = target_agent.predict(node.state['World State'])
        eval_list.append((node.state['World State'],
            sum(-1 * probability * log(probability) for probability in predicted_actions.values() if probability > 0)))

    traverse_graph_topologically(map_graph_by_depth(policy_root), evaluate)
    return eval_list


def weighted_local_info_entropy(policy_root, target_agent, prune_fn):
    """
    Weight by lambda^t until we can figure out a better estimate of the probabilities involved in:
        Information Entropy           λ^t sum_{p(a)} [ p(a) log p(a) ]
    """
    pass


def local_value_of_information(policy_root, target_agent, prune_fn):
    """
    Value of Information            sum_{p(a)} [ V(s | a, π') - V(s | a, π)]
    """
    eval_list = []

    def evaluate(node):
        if prune_fn(node):
            return

        # Calculate entropy
        predicted_actions = target_agent.predict(node.state['World State'])

        # needs identity
        old_action_values = individual_agent_action_values(agent, node.state['World State'], node.state['Models'], node.action_space,
                                                         node.action_values())
        old_policy_action = greedy_action(node)

        calc_value = 0
        for teammate_action, probability in predicted_actions.items():
            # update model

            # Q(s,a) with model updated and policy changed
            new_policy_action = None
            state_value_new_policy = 0

            # Q(s,a) with model updated and policy unchanged
            state_value_old_policy = 0

        eval_list.append((node.state['World State'],
            sum(probability * () for action, probability in predicted_actions.items() if probability > 0)))

    traverse_graph_topologically(map_graph_by_depth(policy_root), evaluate)
    return eval_list


def weighted_val_of_information(policy_root, target_agent, prune_fn):
    """
    Weight by lambda^t until we can figure out a better estimate of the probabilities involved in:
        Value of Information            λ^t sum_{p(a)} [ V(s | a, π') - V(s | a, π)]
    """

    pass


def local_absolute_error(policy_root, target_agent, prune_fn):
    """
    Absolute Error                  sum_{p(a)} [ V(s | a) - V(s)]
    """
    pass


def weighted_absolute_error(policy_root, target_agent, prune_fn):
    """
    Weight by lambda^t until we can figure out a better estimate of the probabilities involved in:
        Absolute Error                  sum_{p(a)} [ V(s | a) - V(s)]
    """
    pass


def local_utility_variance(policy_root, target_agent, prune_fn):
    """
    Utility Variance                sum_{p(a)} [ V(s | a) - V(s)]^2
    """
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
            agent_action_expectations[joint_action[target_agent]] += joint_action_prob * joint_action_values[
                joint_action]

        return sum(action_prob * (agent_action_expectations[action] - node.future_value) ** 2
                   for action, action_prob in model_predictions[target_agent].items())
    pass


def weighted_utility_variance(policy_root, target_agent, prune_fn):
    """
    Weight by lambda^t until we can figure out a better estimate of the probabilities involved in:
        Utility Variance                sum_{p(a)} [ V(s | a) - V(s)]^2
    """
    pass


def random(root, agent_identity):
    """
    Give random evaluations, as a comparison baseline.
    """
    pass


def most_likely_next_state(root, agent_identity):
    """
    Order states by most likely - p(s | π)
    """
    pass


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


