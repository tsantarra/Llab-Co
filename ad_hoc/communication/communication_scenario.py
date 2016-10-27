"""
The communication process for learning a teammate's intentions in an multiagent MDP can be viewed itself as an MDP,
just one that is exponentially larger than that of the original problem. The idea here is to construct a communication
policy, that is, a series of intention queries that will eventually shape the primary agent's own action policy.
"""
from ad_hoc.modeling_agent import get_max_action
from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import map_tree, search, greedy_action
from mdp.action import Action, JointActionSpace

from functools import reduce
from operator import mul
from collections import defaultdict, namedtuple

Query = namedtuple('Query', ['agent', 'state'])


class CommScenario:
    def __init__(self, policy_graph, initial_model_state, identity, comm_cost=0):
        """
        Args:
            policy_graph:           The policy graph for the agent whose policy may be modified by obtaining info.
            initial_model_state:    The current state of the agent's model of the other agents involved.
            identity:               The name of the agent.
            comm_cost:              The amount of resources given up by the agent to communicate with another agent.
        """
        # Save current policy graph for future reference
        self._policy_root = policy_graph
        self._state_graph_map = map_tree(self._policy_root)
        self.agent_identity = identity

        # Caches of commonly computed items
        self._policy_cache = {}
        self._model_state_cache = {}

        # Helpful references
        self.previous_queries = {Query(name, comm_state): action for name, model in initial_model_state.items()
                                 for comm_state, action in model.previous_communications.items()}
        self._all_world_states_in_graph = set(state['World State'] for state in self._state_graph_map)
        self._teammates_models_state = initial_model_state  # current model of teammate behavior
        self.comm_cost = comm_cost  # Set cost per instance of communication

    def initial_state(self):
        """
        The communication state is simply a list of queries and responses along with an 'End?' flag, indicating the
        agent has voluntarily ended the communication process.
        """
        return State({'Queries': State(self.previous_queries), 'End?': False})

    def actions(self, policy_state):
        """
        Each query action available to an agent corresponds with a state in the policy graph. In effect, performing the
        query action is simply asking the other agent what its policy action is for the given state. In order to reduce
        the space of queried states, there are some criteria for pruning states:

            - The scenario is terminal. It does not matter what any agent's policy is at such a state.

            - The state has already been queried. The agent already has the necessary information.

            - The state is not reachable. Given the current information of the other agents' policies, some states may
            be excluded as there is a 0% likelihood of being in the state in the future.

            - The other agent only has one available action. There is no uncertainty about its eventual action.

        The agent may additionally 'Halt' the communication process. This is a vital addition, as in scenarios with a
        cost associated with each communcative action, the agent must minimize the number of queries it poses.
        """
        # No actions for halted case.
        if policy_state['End?']:
            return set()

        # Compute or retrieve model state from communicated policy info
        model_state = self._get_model_state(policy_state)

        # Calculate reachable states
        reachable_nodes = set()
        compute_reachable_nodes(self._policy_root, reachable_nodes, model_state)
        reachable_states = set(node.state['World State'] for node in reachable_nodes
                               if not node.scenario_end)  # Automatically filter out end states

        # Construct the action set over all other agents
        action_set = set()
        for agent_name in model_state:
            possible_queries = reachable_states \
                               - set(query.state for query in policy_state['Queries'] if query.agent == agent_name)

            action_set |= set(Action({self.agent_identity: Query(agent_name, state)})
                              for state in possible_queries)

        action_set.add(Action({self.agent_identity: 'Halt'}))
        return action_set

    def transition(self, policy_state, action):
        """
        For a given query, consider the potential reponses from the modeled agent. Then construct a new policy state
        for each possibility.
        """
        # Get the agent's action
        query_action = action[self.agent_identity]

        # Termination action
        if query_action == 'Halt':
            return Distribution({policy_state.update({'End?': True}): 1.0})

        # The transition probabilities are dependent on the model formed by all current queries.
        target_agent_name, query_world_state = query_action
        model_state = self._get_model_state(policy_state)
        predicted_responses = model_state[target_agent_name].predict(query_world_state)

        # Consider all possible results
        resulting_states = Distribution()
        for target_agent_action, probability in predicted_responses.items():
            # Add query to current queries (as a copy)
            new_query_set = policy_state['Queries'].update({query_action: target_agent_action})
            new_policy_state = policy_state.update({'Queries': new_query_set})

            # Construct new policy state
            resulting_states[new_policy_state] = probability

        # Sanity check.
        assert abs(sum(resulting_states.values()) - 1.0) < 10e-6, \
            'Predicted query action distribution does not sum to 1.'

        return resulting_states

    def end(self, policy_state):
        """
        Returns True if given the current policy state, there is no incentive to communicate further.

        Case 1: All reachable states have been queried.

        Case 2: We can bound the amount of util to be gained from further communication. If that potential is less
                than the cost of communication, there is no incentive to continue.
                - Bounds: max out of policy util - min in policy util
        """
        if policy_state['End?']:
            return True

        model_state = self._get_model_state(policy_state)
        nodes = set()
        compute_reachable_nodes(node=self._policy_root, visited_set=nodes, model_state=model_state)
        reachable_states = set(node.state['World State'] for node in nodes)

        # all teammate reachable states - all queried states
        if all(len(reachable_states - set(query.state for query in policy_state['Queries']
                                          if query.agent == target_agent)) == 0 for target_agent in model_state):
            return True

        # Early termination
        policy = self._get_policy(policy_state)
        root_node = self._policy_root
        agent_next_action = policy[root_node.state]
        action_space = root_node.action_space
        agent_available_actions = action_space.individual_actions(self.agent_identity)

        # Policy commitments
        policy_commitments = defaultdict(dict)
        for query, action in policy_state['Queries'].items():
            policy_commitments[query.state][query.agent] = [action]

        root_constraints = policy_commitments[root_node.state] if root_node.state in policy_commitments else {}

        # Safety value - Expectation over possible resulting states from all actions that fit the agent's policy
        min_in_policy_node_values = {}
        min_exp_util_fixed_policy(node=root_node,
                                  node_values=min_in_policy_node_values,
                                  agent_identity=self.agent_identity,
                                  policy=policy,
                                  policy_commitments=policy_commitments)

        safety_value = min_in_policy_node_values[root_node]

        # For all non-policy actions at the current node, consider the best possible return.
        # Add in agent's non-policy next action
        policy_commitments[root_node.state][self.agent_identity] = agent_available_actions - set([agent_next_action])

        max_out_of_policy_node_values = {}
        max_exp_util_free_policy(node=root_node,
                                 node_values=max_out_of_policy_node_values,
                                 policy_commitments=policy_commitments)

        optimistic_value = max_out_of_policy_node_values[root_node]

        if optimistic_value - safety_value <= self.comm_cost:
            print('Early termination! {opt} {safe} {cost}'.format(opt=optimistic_value,
                                                                  safe=safety_value,
                                                                  cost=self.comm_cost))
            return True
        else:
            return False

    def utility(self, old_policy_state, action, new_policy_state):
        """
        The expected utility for a single query is given as the change in expected utilities for policies before and
        after the query.
        """
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
        """
        To save on computation, we have cached model states corresponding to policy states.
        """
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
        """
        To save on computation, we have cached agent policies by policy/information states. When a policy hasn't been
        computed for a given policy state, compute one with a graph traversal.
        """

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
        """
        Calculate the expected utility of a given policy.
        """

        def use_precomputed_policy(node, action_values, policy):
            action = policy[node.state]
            return action_values[action]

        model_state = self._get_model_state(policy_state)
        return traverse_policy_graph(node=self._policy_root, node_values={}, model_state=model_state,
                                     policy=policy, policy_fn=use_precomputed_policy,
                                     agent_identity=self.agent_identity)


def communicate(agent, agent_dict, passes):
    """
    This is the primary method for initiating a series of policy queries. It is run as any other scenario:
        - Initiate scenario
        - Plan via graph search
        - Traverse through graph, querying the other agents
        - Recalculate agent models and the primary agent's policy
    """
    # Initialize scenario
    comm_scenario = CommScenario(policy_graph=agent.policy_graph_root,
                                 initial_model_state=agent.policy_graph_root.state['Models'],
                                 identity=agent.identity,
                                 comm_cost=0)

    print('BEGIN COMMUNICATION/// ORIGINAL ACTION: ', get_max_action(agent.policy_graph_root, agent.identity))
    # Complete graph search
    comm_graph_node = search(state=comm_scenario.initial_state(),
                             scenario=comm_scenario,
                             iterations=passes,
                             heuristic=lambda comm_state: 0,
                             view=True)

    action = greedy_action(comm_graph_node)
    query_action = action[agent.identity]

    # Initial communication options
    current_policy_state = comm_graph_node.state

    while not current_policy_state['End?']:
        # Check for termination
        if query_action == 'Halt':
            print('Halt')
            break

        # Response
        query_target = agent_dict[query_action.agent]
        response = query_target.get_action(query_action.state)
        print('Query:\n{state}\nResponse: {response}'.format(state=query_action.state, response=response))

        # Update position in policy state graph
        new_query_state = current_policy_state['Queries'].update({query_action: response})
        new_policy_state = current_policy_state.update({'Queries': new_query_state})
        comm_graph_node = comm_graph_node.find_matching_successor(new_policy_state, action=action)

        # Check if graph ends
        if not comm_graph_node.successors:
            print('Reached end of comm graph.')
            break

        # Calculate next step
        action = greedy_action(comm_graph_node)
        query_action = action[agent.identity]
        current_policy_state = comm_graph_node.state

    # update model
    queries = comm_graph_node.state['Queries'].items()
    for agent_name in [name for name in agent_dict if name != 'Agent']:
        query_state_action_pairs = [(query.state, response) for query, response in queries if query.agent == agent_name]
        new_model = agent.model_state[agent_name].communicated_policy_update(query_state_action_pairs)
        agent.model_state = agent.model_state.update({agent_name: new_model})

    new_root_state = agent.policy_graph_root.state.update({'Models': agent.model_state})
    agent.update_policy_graph(agent.policy_graph_root, new_root_state)

    action = get_max_action(agent.policy_graph_root, agent_name)
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
    individual_agent_actions = node.action_space.individual_actions()
    world_state = node.state['World State']
    resulting_models = {agent_name:
                            {action: model_state[agent_name].update(world_state, action) for action in
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
            resulting_node_value = traverse_policy_graph(resulting_state_node, node_values, new_model_state, policy,
                                                         policy_fn, agent_identity)

            joint_action_values[joint_action] += result_probability * \
                                                 (node.successor_transition_values[(
                                                     resulting_state_node.state, joint_action)] + resulting_node_value)

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


def compute_reachable_nodes(node, visited_set, model_state):
    """ Fills a given set of nodes with all unique nodes in the subtree. """
    visited_set.add(node)

    if not node.successors:
        return

    world_state = node.state['World State']
    predicted_actions = {other_agent: set(action for action, prob in other_agent_model.predict(world_state).items()
                                          if prob != 0.0)
                         for other_agent, other_agent_model in model_state.items()}

    resulting_models = {agent_name:
                            {action: model_state[agent_name].update(world_state, action) for action in
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
        action: sum(node.successor_transition_values[(successor.state, action)] + node_values[successor]
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
    action_values = {
        action: sum(node.successor_transition_values[(successor.state, action)] + node_values[successor]
                    for successor, probability in node.successors[action].items())
        for action in action_space}

    node_values[node] = max(action_values.values())
