"""
The communication process for learning a teammate's intentions in an multiagent MDP can be viewed itself as an MDP,
just one that is exponentially larger than that of the original problem. The idea here is to construct a communication
policy, that is, a series of teammate policy queries that will eventually shape the primary agent's own action policy.
"""
from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import map_graph, search, greedy_action
from mdp.action import Action
from agents.modeling_agent import get_max_action, single_agent_policy_backup
from agents.communication.graph_utilities import map_graph_by_depth, traverse_graph_topologically, \
    compute_reachable_nodes, recursive_traverse_policy_graph

from collections import namedtuple
from math import inf as infinity
from heapq import nlargest
from copy import deepcopy

Query = namedtuple('Query', ['agent', 'state'])


class CommScenario:
    def __init__(self, policy_graph, initial_model_state, identity, evaluate_query_fn=None, max_branching_factor=infinity, comm_cost=0, policy_backup_op=single_agent_policy_backup()):
        """
        Args:
            policy_graph:           The policy graph for the agent whose policy may be modified by obtaining info.
            initial_model_state:    The current state of the agent's model of the other agents involved.
            identity:               The name of the agent.
            comm_cost:              The amount of resources given up by the agent to communicate with another agent.
        """
        # Save current policy graph for future reference
        self._agent_identity = identity
        self._policy_root = deepcopy(policy_graph)
        self._teammates_models_state = initial_model_state  # Current model of teammate behavior
        self._evaluate_node_queries_fn = evaluate_query_fn if evaluate_query_fn else (lambda *args: 0)
        self._max_branches = max_branching_factor
        self._policy_backup_op = policy_backup_op
        self.comm_cost = comm_cost  # Cost per instance of communication

        # References to policy graph
        self._state_graph_map = map_graph(self._policy_root)

        # Helpful references
        self.initial_state_queries = State({Query(name, comm_state): action for name, model in initial_model_state.items()
                              for comm_state, action in model.previous_communications.items()})
        self._initial_policy_state = State({'Queries': State(self.initial_state_queries), 'End?': False})

        # Caches of commonly computed items
        self._policy_cache = {}
        self._model_state_cache = {self._initial_policy_state: self._teammates_models_state}
        self._policy_graph_cache = {self._initial_policy_state: self._policy_root}

    def initial_state(self):
        """
        The communication state is simply a list of queries and responses along with an 'End?' flag, indicating the
        agent has voluntarily ended the communication process.
        """
        return State({'Queries': State(self.initial_state_queries), 'End?': False})

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

        # Call provided query selector function
        def prune_query(node, target_agent_name):
            return node.scenario_end or \
                (Query(target_agent_name, node.state['World State']) in policy_state['Queries']) or \
                (len(node.action_space.individual_actions(target_agent_name)) <= 1)

        # Need to consider all teammates involved.
        query_evaluations = []
        policy_graph = self._get_policy_graph(policy_state)
        for target_agent in self._teammates_models_state:
            query_evaluations.extend(self._evaluate_node_queries_fn(policy_graph, target_agent, prune_query))

        # Ensure that nlargest actually keeps unique queries, and not multipe of the same query,
        # resulting from different nodes having different evaluations (due to different models)
        action_set = set(Action({self._agent_identity: query_val[0]}) for query_val in
                         nlargest(self._max_branches, query_evaluations.items(), key=lambda qv: qv[1]))

        # Add 'Halt" action for terminating queries
        action_set.add(Action({self._agent_identity: 'Halt'}))
        return action_set

    def transition(self, policy_state, action):
        """
        For a given query, consider the potential reponses from the modeled agent. Then construct a new policy state
        for each possibility.
        """
        # Get the agent's action
        query_action = action[self._agent_identity]

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

        return False

    def utility(self, old_policy_state, action, new_policy_state):
        """
        The expected utility for a single query is given as the change in expected utilities for policies before and
        after the query.
        """
        # If the action is to stop communicating, there is no further utility gain.
        if action[self._agent_identity] == 'Halt':
            return 0

        # Calculate old policies
        old_policy = self._get_policy(old_policy_state)
        new_policy = self._get_policy(new_policy_state)

        # Calculate expected util under new policy state for both old and new policies
        exp_util_old_policy = self._get_policy_ev(policy_state=new_policy_state, policy=old_policy)
        exp_util_new_policy = self._get_policy_ev(policy_state=new_policy_state, policy=new_policy)

        return exp_util_new_policy - exp_util_old_policy - self.comm_cost

    def _get_policy_graph(self, policy_state, old_policy_state=None):
        """
        Rather than save each bit of information separately (policy, EV, model state, etc), just keep
        a single copy of the policy graph for each policy state. A new policy calculation is needed
        for each step of the communicative search model anyway.
        """
        # Early exit: If we have already calculated this new policy graph, return the cached version.
        if policy_state in self._policy_graph_cache:
            return self._policy_graph_cache[policy_state]

        # If no reference policy state is given, use the initial one.
        if old_policy_state is None:
            old_policy_state = self._initial_policy_state

        # Copy the most similar graph (from a similar policy state).
        new_root = deepcopy(self._policy_graph_cache[old_policy_state])
        keys_diff = policy_state['Queries'].keys() - old_policy_state['Queries'].keys()
        assert len(keys_diff) <= 1, 'Incorrect number of keys in keys_diff between successive policy states.'

        # Eary exit: If no difference in keys, the scenario action ending the comm sequence was selected.
        if not keys_diff:
            self._policy_graph_cache[policy_state] = self._policy_graph_cache[old_policy_state]
            return self._policy_graph_cache[policy_state]

        # First, update each node (order does not matter).
        depth_map = map_graph_by_depth(new_root)
        for node in depth_map:
            # Update just the model corresponding to the query.
            model_state = node.state['Models']
            for query in keys_diff:
                response = policy_state['Queries'][query]
                model_state = model_state.update({query.agent: model_state[query.agent].update(query.state, response)})
            node.state = node.state.update({'Models': model_state})

        # With new models set, recalculate the policy. This must be done bottom-up.
        def update_graph(graph_node, horizon):
            self._policy_backup_op(graph_node, self._agent_identity)

        traverse_graph_topologically(depth_map, update_graph, top_down=False)

        # Store and return new policy graph.
        self._policy_graph_cache[policy_state] = new_root

        return new_root

    def _get_model_state(self, policy_state):
        """
        To save on computation, we have cached model states corresponding to policy states.
        """
        if policy_state in self._model_state_cache:
            return self._model_state_cache[policy_state]

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
            _ = recursive_traverse_policy_graph(node=self._policy_root, node_values={}, model_state=model_state,
                                                policy=policy, policy_fn=compute_policy,
                                                agent_identity=self._agent_identity)
            self._policy_cache[policy_state] = policy
            return policy

    def _get_policy_ev(self, policy_state, policy):
        """
        Calculate the expected utility of a given policy.
        """
        def use_precomputed_policy(node, action_values, policy):
            return action_values[policy[node.state]]

        model_state = self._get_model_state(policy_state)
        return recursive_traverse_policy_graph(node=self._policy_root, node_values={}, model_state=model_state,
                                               policy=policy, policy_fn=use_precomputed_policy,
                                               agent_identity=self._agent_identity)


def communicate(agent, agent_dict, passes, comm_cost=0):
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
                                 comm_cost=comm_cost) # backup op?!

    original_action = get_max_action(agent.policy_graph_root, agent.identity)
    print('BEGIN COMMUNICATION/// ORIGINAL ACTION: ', original_action)

    # Complete graph search
    comm_graph_node = search(state=comm_scenario.initial_state(),
                             scenario=comm_scenario,
                             iterations=passes,
                             heuristic=lambda comm_state: 0)

    # No communication can help or no communication possible.
    if not comm_graph_node.successors:
        return original_action

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

    action = get_max_action(agent.policy_graph_root, agent.identity)
    print('END COMMUNICATION/// NEW ACTION:', action)

    return action


