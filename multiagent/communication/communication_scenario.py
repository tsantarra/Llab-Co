from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import map_tree

from collections import defaultdict


def calculate_policy(node, node_values, policy, compute_new_policy=True):
    """ Computes a policy maximizing expected utility given a probabilistic agent model for other agents. """
    if node in node_values:
        # Already visited this node. Return its pre-computed value.
        return node_values[node]

    # Otherwise, compute the node's EV
    node_values[node] = node.immediate_value
    if node.successors:
        # Calculate expected return for each action at the given node
        action_values = defaultdict(float)
        for action, child_distribution in node.successors.items():
            action_values[action] = sum(prob * calculate_policy(child_node, node_values, policy, compute_new_policy)
                                        for child_node, prob in child_distribution.items()) \
                                    / sum(child_distribution.values())

        agent_turn = node.state['Turn']
        if agent_turn not in node.state['Models']:  # Agent maximized expectation
            if not compute_new_policy:
                # Given a pre-computed policy
                action = policy[node.state]
                node_values[node] += action_values[action]
            else:
                # Computing a new policy
                action, action_value = max(action_values.items(), key=lambda pair: pair[1])
                policy[node.state] = action
                node_values[node] += action_value
        else:  # Agent predicts action distribution and resulting expected value
            action_distribution = node.state['Models'][agent_turn].predict(node.state)
            node_values[node] += action_distribution.expectation(action_values, require_exact_keys=False)

    return node_values[node]


def reachable_nodes(node, visited_set):
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    visited_set.add(node)

    if not node.successors:
        return

    acting_agent = node.state['Turn']
    if acting_agent not in node.state['Models']:
        # This is the modeling agent's turn
        successor_set = set(successor_node for dist in node.successors.values()
                            for successor_node in dist if successor_node.state not in visited_set)

    else:
        # Modeled agent's turn
        model = node.state['Models'][acting_agent]
        possible_actions = [action for action, prob in model.predict(node.state).items() if prob > 0.0]
        successor_set = set(successor_node for action in possible_actions for dist in node.successors[action]
                            for successor_node in dist if successor_node not in visited_set)

    for successor, args in successor_set:
        reachable_nodes(successor, visited_set)


def min_in_policy(node, visited_set, terminal_set, policy):  # TODO Not vital, but possibly good to have.
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    # GOT IT. FIX POLICY, minimize teammate choices!
    pass


def max_out_of_policy(node, visited_set, policy, in_or_out_flag):  # TODO Not vital, but possibly good to have.
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    # Modified Viterbi algorithm might handle this. Need some way to flag if a result is in or out of policy.
    pass


class CommScenario:
    def __init__(self, policy_graph, initial_model, comm_cost=0):
        # Save current policy graph for future reference
        self._policy_root = policy_graph
        self._state_graph_map = map_tree(self._policy_root)

        self._policy_ev_cache = {}  # To avoid recalculating policies?
        self._base_policy = {}  # the original policy
        self._modeler_states = set()  # states where the policy may change
        self._teammate_states = set()  # states to query
        self._teammate_model = initial_model  # current model of teammate behavior
        self.comm_cost = comm_cost  # Set cost per instance of communication

    def _calculate_ev(self, policy_state):
        """
        Given a current policy state, return the expected utility from the current world state.
        """
        # If we have already calculated this, return the cached value.
        if policy_state in self._policy_ev_cache:
            return self._policy_ev_cache[policy_state]

        # Otherwise, calculate via graph traversal.
        pass

    @staticmethod
    def initial_state():
        return State()

    def actions(self, policy_state):  # TODO reachability
        return self._teammate_states - policy_state.keys()

    def transition(self, policy_state, query_state):
        # The transition probabilities are dependent on the model formed by all current queries.
        model = self._teammate_model.copy()
        model = model.communicated_policy_update(policy_state.items())
        predicted_responses = model.predict(query_state)

        resulting_states = Distribution()
        for action, probability in predicted_responses.items():
            # Add query to current queries (as a copy)
            new_policy_state = policy_state.update((query_state, action))

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

        nodes = set()
        reachable_nodes(self._policy_root, nodes)
        reachable_states = set(node.state for node in nodes)

        # all teammate reachable states - all queried states
        return len((self._teammate_states & reachable_states) - policy_state.keys()) > 0

    def utility(self, old_policy_state, query, new_policy_state):
        # TODO  More nuanced than this. It's new EV - EV of new info under OLD POLICY
        response = new_policy_state[query]
        return self._calculate_ev(new_policy_state) - self._calculate_ev(old_policy_state) - self.comm_cost
