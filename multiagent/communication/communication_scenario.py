from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import map_tree, search, greedy_action
from mdp.action import Action

from functools import reduce
from operator import mul
from collections import defaultdict, namedtuple

Query = namedtuple('Query', ['agent', 'state'])


class CommScenario:
    def __init__(self, policy_graph, initial_model_state, identity, comm_cost=0):
        # Save current policy graph for future reference
        self._policy_root = policy_graph
        self._state_graph_map = map_tree(self._policy_root)
        self.agent_identity = identity

        # Caches of commonly computed items
        self._policy_cache = {}
        self._model_state_cache = {}

        # Helpful references
        self.previous_queries = {name: set(comm_state
                                           for comm_state in model.previous_communications)
                                 for name, model in initial_model_state.items()}
        self._all_world_states_in_graph = set(state['World State'] for state in self._state_graph_map)
        self._teammates_models_state = initial_model_state  # current model of teammate behavior
        self.comm_cost = comm_cost  # Set cost per instance of communication

    def initial_state(self):
        return State({'Queries': State(), 'End?': False})

    def actions(self, policy_state):
        # No actions for halted case.
        if policy_state['End?']:
            return set()

        # Compute or retrieve model state from communicated policy info
        model_state = self._get_model_state(policy_state)

        # Calculate reachable states
        reachable_nodes = set()
        compute_reachable_nodes(self._policy_root, reachable_nodes, model_state)
        reachable_states = set(node.state['World State'] for node in reachable_nodes)

        action_set = set()
        for agent_name in model_state:
            possible_query_states = reachable_states - set(query.state for query in policy_state['Queries']
                                                           if query.agent == agent_name) \
                                    - self.previous_queries[agent_name]

            action_set |= set(Action({self.agent_identity: Query(agent_name, state)})
                              for state in possible_query_states)

        action_set.add(Action({self.agent_identity: 'Halt'}))
        return action_set

    def transition(self, policy_state, query_action):
        # Termination action TODO WRONG SINCE SIMULTANEOUS UPDATE
        if query_action[self.agent_identity] == 'Halt':
            return Distribution({policy_state.update({'End?': True}): 1.0})

        # The transition probabilities are dependent on the model formed by all current queries.
        agent_name, query_world_state = query_action
        model_state = self._get_model_state(policy_state)
        predicted_responses = model_state[agent_name].predict(query_world_state)

        resulting_states = Distribution()
        for agent_action, probability in predicted_responses.items():
            # Add query to current queries (as a copy)
            new_query_state = policy_state['Queries'].update({query_action: agent_action})
            new_policy_state = policy_state.update({'Queries': new_query_state})

            # Construct new policy state
            resulting_states[new_policy_state] = probability

        # assert policy_state not in resulting_states, 'Accidental self parent. ' + str(policy_state)

        assert abs(sum(resulting_states.values()) - 1.0) < 10e-6, \
            'Predicted query action distribution does not sum to 1.'

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
        reachable_states = set(node.state['World State'] for node in nodes)

        # all teammate reachable states - all queried states
        return len(reachable_states
                   - set(query.state for query in policy_state['Queries'])
                   - set(comm_state for comm_set in self.previous_queries.values()
                         for comm_state in comm_set)) == 0

    def utility(self, old_policy_state, action, new_policy_state):
        # If the action is to stop communicating, there is no further utility gain.
        if action[self.agent_identity] == 'Halt':
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
                policy_pairs = [(query, action) for query, action in policy_state['Queries'].items()
                                if query.agent == name]
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
                                                  policy=policy, policy_fn=compute_policy,
                                                  agent_identity=self.agent_identity)
            self._policy_cache[policy_state] = policy
            return policy

    def _get_policy_ev(self, policy_state, policy):
        def use_precomputed_policy(node, action_values, policy):
            action = policy[node.state]
            return action_values[action]

        model_state = self._get_model_state(policy_state)
        return traverse_policy_graph(node=self._policy_root, node_values={}, model_state=model_state,
                                     policy=policy, policy_fn=use_precomputed_policy,
                                     agent_identity=self.agent_identity)


def communicate(state, agent, agent_dict, passes):
    # Initialize scenario
    comm_scenario = CommScenario(policy_graph=agent.policy_graph_root,
                                 initial_model_state=agent.policy_graph_root.state['Models'],
                                 comm_cost=0)

    print('BEGIN COMMUNICATION')
    # Complete graph search
    (query_action, comm_graph_node) = search(state=comm_scenario.initial_state(),
                                             scenario=comm_scenario,
                                             iterations=passes,
                                             heuristic=lambda comm_state: 0,
                                             view=True)

    from visualization.graph import show_graph
    if comm_graph_node.successors:
        show_graph(comm_graph_node, skip_cross_optimization=True)

    # Initial communication options
    current_policy_state = comm_graph_node.state

    while not current_policy_state['End?']:
        # Check for termination
        if query_action[agent.identity] == 'Halt':
            print('Halt')
            break

        # response
        query_target = agent_dict[query_action.name]
        response = query_target.get_action(query_action)

        # update position in policy state graph
        new_query_state = current_policy_state['Queries'].update({query_action: response})
        new_policy_state = current_policy_state.update({'Queries': new_query_state})

        print('Query:', query_action)
        print('Response:', response)
        for stac, val in comm_graph_node.successor_transition_values.items():
            print('policy state\n', stac[0])
            print('query state\n', stac[1])
            print('val:', val)
        print('Util:', comm_graph_node.successor_transition_values[(new_policy_state, query_action)])

        comm_graph_node = comm_graph_node.find_matching_successor(new_policy_state, action=query_action)

        # calculate next step
        query_action = greedy_action(comm_graph_node)[agent.identity]
        current_policy_state = comm_graph_node.state

    # update model
    queries = comm_graph_node.state['Queries'].items()
    for agent_name in [name for name in agent_dict if name != 'Agent']:
        agent_queries = [(query, response) for query, response in queries if query.name == agent_name]
        new_model = agent.model_state[agent_name].communicated_policy_update(agent_queries)
        agent.model_state = agent.model_state.update({agent_name: new_model})

    new_root_state = agent.policy_graph_root.state.update({'Models': agent.model_state})
    agent.update_policy_graph(agent.policy_graph_root, new_root_state)

    action = agent.get_action(state)
    print('END COMMUNICATION/// NEW ACTION:', action)

    return action


def traverse_policy_graph(node, node_values, model_state, policy, policy_fn, agent_identity):
    """ Computes a policy maximizing expected utility given a probabilistic agent model for other agents. """
    # Already visited this node. Return its computed value.
    if node in node_values:
        return node_values[node]

    # Leaf node. Value already calculated as immediate + heuristic.
    if not node.successors:
        node_values[node] = node.future_value
        return node.future_value

    # Update all individual agent models
    individual_agent_actions = node.individual_agent_actions
    world_state = node.state['World State']
    resulting_models = {agent_name:
                            {action: model_state[agent_name].update(world_state, action) for action in
                             agent_actions}
                        for agent_name, agent_actions in individual_agent_actions.items()}

    # Calculate expected return for each action at the given node
    joint_action_values = defaultdict(float)
    for joint_action, result_distribution in node.successors.items():
        assert abs(sum(result_distribution.values()) - 1.0) < 10e-5, 'Action probabilities do not sum to 1.'

        # Construct new model state from individual agent models
        new_model_state = State({agent_name: resulting_models[agent_name][joint_action[agent_name]]
                                 for agent_name in model_state})

        # Traverse to successor nodes
        for resulting_state_node, result_probability in result_distribution.items():
            resulting_node_value = traverse_policy_graph(resulting_state_node, node_values, new_model_state, policy,
                                                         policy_fn, agent_identity)

            joint_action_values[joint_action] += result_probability * \
                                                 (node.successor_transition_values[(
                                                 resulting_state_node.state, joint_action)] + resulting_node_value)

    # Now breakdown joint actions so we can calculate the primary agent's action values
    agent_individual_actions = node.individual_agent_actions
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


def compute_reachable_nodes(node, visited_set, model_state):
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    visited_set.add(node)

    if not node.successors:
        return

    world_state = node.state['World State']
    ruled_out_actions = {other_agent: set(action for action, prob in other_agent_model.predict(world_state).items()
                                          if prob == 0.0)
                         for other_agent, other_agent_model in model_state.items()}

    # Filter out - considers all existing joint actions. Probably faster to construct a new set of joint actions.
    # joint_actions = [joint_action for joint_action in node.successors
    #                 if all(joint_action[agent] not in ruled_out_actions[agent] for agent in model_state)]

    individual_actions = {agent_name: action_set - ruled_out_actions[agent_name]
                          for agent_name, action_set in node.individual_agent_actions.items()}

    resulting_models = {agent_name:
                            {action: model_state[agent_name].update(world_state, action) for action in
                             agent_actions}
                        for agent_name, agent_actions in individual_actions.items()}

    for joint_action in Action.all_joint_actions(individual_actions):
        # Update model state
        new_model_state = State({agent_name: resulting_models[agent_name][joint_action[agent_name]]
                                 for agent_name in model_state})

        # Traverse through applicable successor nodes
        for successor_node in (successor for successor in node.successors[joint_action] if
                               successor not in visited_set):
            compute_reachable_nodes(successor_node, visited_set, new_model_state)


################################################# In progress #####################################################

def min_exp_util(node, node_values, policy):
    """ Searching for minimum expected utility within the given policy. """
    # Already covered this node and subgraph
    if node in node_values:
        return

    # Leaf node. Simply the node's future value.
    if not node.successors:
        node_values[node] = node.future_value

    # Recurse through subgraph.
    for successor in node.successor_set():
        min_exp_util(successor, node_values, policy)

    # Calculate new minimum over new node values.




def max_exp_util(node, node_values, policy):
    """ Search for maximum expected utility given one or more policy changes. """

    # Note: due to the online nature of comm planning, we need only find the max of outcome distributions over changes
    # in the root's policy, not all possible policy changes.
    pass
