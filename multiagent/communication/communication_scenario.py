from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import map_tree, search, _greedy_action

from collections import defaultdict


class CommScenario:
    def __init__(self, policy_graph, initial_model_state, comm_cost=0):
        # Save current policy graph for future reference
        self._policy_root = policy_graph
        self._state_graph_map = map_tree(self._policy_root)

        # Caches of commonly computed items
        self._policy_cache = {}
        self._policy_ev_cache = {}
        self._model_state_cache = {}

        # Helpful references
        self._teammate_states = set(state for state in self._state_graph_map if state['World State']['Turn'] in initial_model_state)
        self._teammates_models_state = initial_model_state  # current model of teammate behavior
        self.comm_cost = comm_cost  # Set cost per instance of communication

    def initial_state(self):
        return State({'Queries': State(), 'End?': False})

    def actions(self, policy_state):
        # Compute or retrieve model state from communicated policy info
        model_state = self._get_model_state(policy_state)

        # Calculate reachable states
        reachable_nodes = set()
        compute_reachable_nodes(self._policy_root, reachable_nodes, model_state)
        reachable_states = set(node.state for node in reachable_nodes)

        return ((self._teammate_states & reachable_states) - policy_state['Queries'].keys()).add('Halt')

    def transition(self, policy_state, action):
        # Termination action
        if action == 'Halt':
            return policy_state.update({'End?': True})

        # The transition probabilities are dependent on the model formed by all current queries.
        query_state = action
        agent_name = query_state['Turn']
        model_state = self._get_model_state(policy_state)
        predicted_responses = model_state[agent_name].predict(query_state)

        resulting_states = Distribution()
        for agent_action, probability in predicted_responses.items():
            # Add query to current queries (as a copy)
            new_query_state = policy_state['Queries'].update({query_state: agent_action})
            new_policy_state = policy_state.update({'Queries': new_query_state})

            # Construct new policy state
            resulting_states[new_policy_state] = probability

        assert abs(sum(resulting_states.values) - 1.0) < 10e-6, 'Predicted query action distribution does not sum to 1.'
        return resulting_states

    def end(self, policy_state):
        """
        Returns True if given the current policy state, there is no incentive to communicate further.

        Case 1: All reachable states have been queried.

        Case 2: Can we bound the total possible remaining util to be gained from communication?
                - Fix policy -> min path -> lower bound for what we're currently going to get
                - Consider all non-policy actions -> max path -> upper bound for changing policy
                - If change max - fixed min < comm cost, no comm would be worthwhile
        """
        if policy_state['End?']:
            return True

        model_state = self._get_model_state(policy_state)

        nodes = set()
        compute_reachable_nodes(node=self._policy_root, visited_set=nodes, model_state=model_state)
        reachable_states = set(node.state for node in nodes)

        # all teammate reachable states - all queried states
        return len((self._teammate_states & reachable_states) - policy_state['Queries'].keys()) > 0

    def utility(self, old_policy_state, action, new_policy_state):
        # If the action is to stop communicating, there is no further utility gain.
        if action == 'Halt':
            return 0

        # Calculate old policies
        old_policy = self._get_policy(old_policy_state)
        new_policy = self._get_policy(new_policy_state)

        # Calculate expected util under new policy state for both old and new policies
        exp_util_old_policy = self._get_policy_ev(policy_state=new_policy_state, policy=old_policy)
        exp_util_new_policy = self._get_policy_ev(policy_state=new_policy_state, policy=new_policy)

        return exp_util_new_policy - exp_util_old_policy - self.comm_cost

    def _get_model_state(self, policy_state):
        if policy_state in self._model_state_cache:
            return self._model_state_cache[policy_state]
        else:
            new_models = {}
            for name, model in self._teammates_models_state.items():
                policy_pairs = [(state, action) for state, action in policy_state['Queries'].items()
                                if state['Turn'] == name]
                new_models[name] = model.communicated_policy_update(policy_pairs)

            model_state = State(new_models)
            self._model_state_cache[policy_state] = model_state
            return model_state

    def _get_policy(self, policy_state):
        def compute_policy(node, action_values, new_policy):
            action, action_value = max(action_values.items(), key=lambda pair: pair[1])
            new_policy[node.state] = action
            return action_value

        if policy_state in self._policy_cache:
            return self._policy_cache[policy_state]
        else:
            policy = {}
            model_state = self._get_model_state(policy_state)
            expected_util = traverse_policy_graph(node=self._policy_root, node_values={}, model_state=model_state,
                                                  policy=policy, policy_fn=compute_policy)
            self._policy_cache[policy_state] = policy
            self._policy_ev_cache[policy_state] = expected_util
            return policy

    def _get_policy_ev(self, policy_state, policy):
        def use_precomputed_policy(node, action_values, policy):
            action = policy[node.state]
            return action_values[action]

        model_state = self._get_model_state(policy_state)
        return traverse_policy_graph(node=self._policy_root, node_values={}, model_state=model_state,
                                     policy=policy, policy_fn=use_precomputed_policy)


def communicate(state, agent, agent_dict):
    # Initialize scenario
    comm_scenario = CommScenario(policy_graph=agent.policy_graph_root,
                                 initial_model_state=agent.policy_graph_root.state['Models'],
                                 comm_cost=0)

    # Complete graph search
    (query_state, comm_graph_node) = search(state=comm_scenario.initial_state(),
                                            scenario=comm_scenario,
                                            iterations=1000,
                                            heuristic=lambda comm_state, scenario: 0)

    # Initial communication options
    current_policy_state = comm_graph_node.state

    while not current_policy_state['End?']:  # TODO WRONG. Myopic lookahead. Need to figure out a way to halt search.
        # response
        query_target = agent_dict[query_state['Turn']]
        response = query_target.get_action(query_state)

        # update position in policy state graph
        new_query_state = current_policy_state['Queries'].update({query_target: response})
        new_policy_state = current_policy_state.update({'Queries': new_query_state})
        comm_graph_node = comm_graph_node.find_matching_successor(new_policy_state, action=query_state)

        # calculate next step
        query_state = _greedy_action(comm_graph_node)
        current_policy_state = comm_graph_node.state

    # update model
    queries = comm_graph_node.state['Queries'].items()
    for agent_name in [name for name in agent_dict if name != 'Agent']:
        agent_queries = [(query, response) for query, response in queries if query['Turn'] == agent_name]
        new_model = agent.model_state[agent_name].communicated_policy_update(agent_queries)
        agent.model_state = agent.model_state.update({agent_name: new_model})

    action = agent.get_action(state)

    return action


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


################################################# In progress #####################################################


def min_in_policy(node, node_values, policy, policy_state):
    """ Searching for minimum expected utility within the given policy. """
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
            action_values[action] += result_probability * min_in_policy(resulting_state_node, node_values,
                                                                        policy, policy_state)

    # Compute the node value
    if active_agent not in node.state['Models']:
        # Given a pre-computed policy, use the associated action
        policy_action = policy[node.state]
        node_values[node] = node.immediate_value + action_values[policy_action]
    else:
        # Other agent
        if node.state['World State'] in policy_state:
            # Already committed to policy action
            node_values[node] = node.immediate_value + action_values[policy_state[node.state['World State']]]
        else:
            # Minimize value (antagonist assumption)
            action, action_value = min(action_values.items(), key=lambda pair: pair[1])
            node_values[node] = node.immediate_value + action_value

    return node_values[node]


def max_out_of_policy(node, node_values_in_pol, node_values_out_of_pol, policy, policy_state):
    """ Search for maximum expected utility given one or more policy changes. """
    # Already visited this node. Return its computed value.
    if node in node_values_in_pol:
        return

    # Leaf node. Value already calculated as immediate + heuristic.
    if not node.successors:
        node_values_in_pol[node] = node.value
        return

    # Calculate expected return for each action at the given node
    for action, result_distribution in node.successors.items():
        assert abs(sum(result_distribution.values()) - 1.0) < 10e-5, 'Action probabilities do not sum to 1.'
        for resulting_state_node, result_probability in result_distribution.items():
            max_out_of_policy(resulting_state_node, node_values_in_pol, node_values_out_of_pol, policy, policy_state)

    active_agent = node.state['Turn']
    # Compute the node value
    if active_agent not in node.state['Models']:
        # Given a pre-computed policy, use the associated action
        policy_action = policy[node.state]
        node_values_in_pol[node] = node.immediate_value + sum(node_values_in_pol[result_node] * prob
                                                              for result_node, prob in
                                                              node.successors[policy_action].items())

        # check max action over max sub-graphs
        max_action_values = defaultdict(float)
        max_action_check = {}
        for action, result_distribution in node.successors.items():
            max_results_in_pol = True
            for result_node, probability in result_distribution.items():
                if node_values_in_pol[result_node] > node_values_out_of_pol[result_node]:
                    max_action_values[action] += node_values_in_pol[result_node] * probability

                else:
                    max_results_in_pol = False
                    max_action_values[action] += node_values_out_of_pol[result_node] * probability

            max_action_check[action] = max_results_in_pol

        max_action, max_action_value = max(max_action_values.items, key=lambda pair: pair[1])

        if max_action == policy[node.state] and max_action_check[max_action]:
            # Best result is in policy. Need to calculate nearest alternative.
            alternative_values = defaultdict(float)
            for action, result_distribution in node.successors.items():
                if not max_action_check[action]:  # out of policy result
                    alternative_values[action] = max_action_values[action]
                else:
                    # Still best with policy. Change a subgraph, minimizing difference.
                    deltas = {result_node:
                                  prob * (node_values_in_pol[result_node] - node_values_out_of_pol[result_node])
                              for result_node, prob in result_distribution.items()}
                    alternative_values[action] = max_action_values[action] - min(deltas.values())

            node_values_out_of_pol[node] = max(alternative_values.values())

        else:
            # Best result is out of policy. All clear.
            node_values_out_of_pol[node] = max_action_value

        return  # End consideration for agent's action.

    # Other agent acting; maximize value (optimist assumption) TODO
    # Still need to pass up policy val and non-policy val?
    # Complicated.

    return
