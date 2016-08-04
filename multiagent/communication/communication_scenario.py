from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import map_tree

from collections import defaultdict


def traverse_policy_graph(node, node_values, model_state, policy, policy_fn):
    """ Computes a policy maximizing expected utility given a probabilistic agent model for other agents. """
    # Already visited this node. Return its computed value.
    if node in node_values:
        return node_values[node]

    # Leaf node. Value already calculated as immediate + heuristic.
    if not node.successors:
        node_values[node] = node.value
        return node.value

    # Calculate expected return for each action at the given node
    active_agent = node.state['Turn']
    action_values = defaultdict(float)
    for action, result_distribution in node.successors.items():
        assert abs(sum(result_distribution.values()) - 1.0) < 10e-5, 'Action probabilities do not sum to 1.'
        for resulting_state_node, result_probability in result_distribution.items():
            if active_agent in model_state:
                new_model = model_state[active_agent].update(node.state, action)
                new_model_state = model_state.update({active_agent: new_model})
            else:
                new_model_state = model_state

            action_values[action] += result_probability * traverse_policy_graph(resulting_state_node, node_values,
                                                                                new_model_state, policy, policy_fn)

    # Compute the node value
    if active_agent not in model_state:
        # Given a pre-computed policy, use the associated action
        node_values[node] = node.immediate_value + policy_fn(node, action_values, policy)
    else:
        # Agent predicts action distribution and resulting expected value
        action_distribution = model_state[active_agent].predict(node.state)
        node_values[node] = node.immediate_value + action_distribution.expectation(action_values,
                                                                                   require_exact_keys=False)

    return node_values[node]


def compute_reachable_nodes(node, visited_set, model_state):
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    visited_set.add(node)

    if not node.successors:
        return

    acting_agent = node.state['Turn']
    if acting_agent not in model_state:
        # This is the modeling agent's turn. All successor nodes are viable.
        for successor_node in (successor for dist in node.successors.values()
                               for successor in dist if successor not in visited_set):
            compute_reachable_nodes(successor_node, visited_set, model_state)
    else:
        # Modeled agent's turn. Prune out actions with 0 probability.
        model = model_state[acting_agent]
        possible_actions = [action for action, prob in model.predict(node.state).items() if prob > 0.0]

        for action in possible_actions:
            for successor_node in (successor for successor in node.successors[action] if successor not in visited_set):
                # Update model state
                new_model = model.update(node.state, action)
                new_model_state = model_state.update({acting_agent: new_model})
                compute_reachable_nodes(successor_node, visited_set, new_model_state)


class CommScenario:
    def __init__(self, policy_graph, initial_model_state, comm_cost=0):
        # Save current policy graph for future reference
        self._policy_root = policy_graph
        self._state_graph_map = map_tree(self._policy_root)

        self._policy_ev_cache = {}  # To avoid recalculating policies?
        self._base_policy = {}  # the original policy
        self._modeler_states = set()  # states where the policy may change
        self._teammate_states = set()  # states to query
        self._teammate_model_state = initial_model_state  # current model of teammate behavior
        self.comm_cost = comm_cost  # Set cost per instance of communication

    def initial_state(self):
        return State()

    def actions(self, policy_state):
        # Update base models with communicated policy info
        model_state = self._teammate_model_state.communicated_policy_update(policy_state.items())

        # Calculate reachable states
        reachable_nodes = set()
        compute_reachable_nodes(self._policy_root, reachable_nodes, model_state)
        reachable_states = set(node.state for node in reachable_nodes)

        return (self._teammate_states & reachable_states) - policy_state.keys()

    def transition(self, policy_state, query_state):
        # The transition probabilities are dependent on the model formed by all current queries.
        agent_name = query_state['Turn']
        base_model = self._teammate_model_state[agent_name].copy()
        model = base_model.communicated_policy_update(policy_state.items())
        predicted_responses = model.predict(query_state)

        resulting_states = Distribution()
        for action, probability in predicted_responses.items():
            # Add query to current queries (as a copy)
            new_policy_state = policy_state.update({query_state: action})

            # Construct new policy state
            resulting_states[new_policy_state] = probability

        assert abs(sum(resulting_states.values) - 1.0) < 10e-6, 'Predicted query action distribution does not sum to 1.'
        return resulting_states

    def end(self, policy_state):
        """
        Returns True if given the current policy state, there is no incentive to communicate further.

        Case 1: All reachable states have been queried.
        SEE CASE 3 -- Case 2: For all remaining states, the expected utility gain is less than or equal to zero.
            - but how would we know unless we exhaustively tried all remaining subsets?
            - if indifferent at all remaining states, is it possible that a pair has the possibility for improvement?
                + No. The change propagated up is 0. No policy is changed. No other states' expected utils change.
            - should cache this calculation so it is not repeated

        Case 3 (?):
            Can we bound the total possible remaining util to be gained from communication?
            I guess if the global max - global min < comm cost, we'd know for sure.
                - It's not global!
                - Fix policy -> min path -> lower bound for what we're currently going to get
                - Consider all non-policy actions -> max path -> upper bound for changing policy
                - If change max - fixed min < comm cost, no comm would be worthwhile
            But what about at the local state level?
            As with Case 2, we should cache this calculation.
            However, Case 2 is a specific instance of Case 3, suggesting we do not need it.
        """
        new_model_state = {}
        for name, model in self._teammate_model_state.items():
            new_model = model.communicated_policy_update(policy_state.items())
            new_model_state[name] = new_model
        new_model_state = State(new_model_state)

        nodes = set()
        compute_reachable_nodes(node=self._policy_root, visited_set=nodes, model_state=new_model_state)
        reachable_states = set(node.state for node in nodes)

        # all teammate reachable states - all queried states
        return len((self._teammate_states & reachable_states) - policy_state.keys()) > 0

    def utility(self, old_policy_state, query_state, new_policy_state):
        # Helper functions
        def compute_policy(node, action_values, policy):
            action, action_value = max(action_values.items(), key=lambda pair: pair[1])
            policy[node.state] = action
            return action_value

        def use_precomputed_policy(node, action_values, policy):
            action = policy[node.state]
            return action_values[action]

        # Calculate old and new model states (for calculating policies)
        old_model_state = {}
        for name, model in self._teammate_model_state.items():
            new_model = model.communicated_policy_update(old_policy_state.items())
            old_model_state[name] = new_model
        old_model_state = State(old_model_state)

        new_model_state = {}
        for name, model in self._teammate_model_state.items():
            new_model = model.communicated_policy_update((query_state, new_policy_state[query_state]))
            new_model_state[name] = new_model
        new_model_state = State(new_model_state)

        # Calculate old policy
        old_policy = {}
        old_expected_util = traverse_policy_graph(node=self._policy_root, node_values={}, model_state=old_model_state,
                                                  policy=old_policy, policy_fn=compute_policy)

        # Calculate new expected util under NEW policy state
        exp_util_old_policy = traverse_policy_graph(node=self._policy_root, node_values={}, model_state=new_model_state,
                                                    policy=old_policy, policy_fn=use_precomputed_policy)

        # Calculate new expected util under NEW POLICY. Return difference.
        new_policy = {}
        exp_util_new_policy = traverse_policy_graph(node=self._policy_root, node_values={}, model_state=new_model_state,
                                                    policy=new_policy, policy_fn=compute_policy)

        return exp_util_new_policy - exp_util_old_policy
