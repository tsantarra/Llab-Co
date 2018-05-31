"""
The communication process for learning a teammate's intentions in an multiagent MDP can be viewed itself as an MDP,
just one that is exponentially larger than that of the original problem. The idea here is to construct a communication
policy, that is, a series of teammate policy queries that will eventually shape the primary agent's own action policy.
"""
from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import map_graph, search, greedy_action
from mdp.action import Action
from agents.modeling_agent import get_max_action, single_agent_policy_backup, individual_agent_action_values
from mdp.graph_utilities import map_graph_by_depth, traverse_graph_topologically, compute_reachable_nodes

from collections import namedtuple
from math import inf as infinity
from heapq import nlargest
from functools import reduce
from operator import mul

Query = namedtuple('Query', ['agent', 'state'])


class CommScenario:
    def __init__(self, policy_graph, initial_model_state, identity, evaluate_query_fn, heuristic,
                 max_branching_factor=infinity, comm_cost=0, policy_backup_op=single_agent_policy_backup):
        """
        Args:
            policy_graph:           The policy graph for the agent whose policy may be modified by obtaining info.
            initial_model_state:    The current state of the agent's model of the other agents involved.
            identity:               The name of the agent.
            comm_cost:              The amount of resources given up by the agent to communicate with another agent.
        """
        # Save current policy graph for future reference
        self._agent_identity = identity
        self._policy_root = policy_graph.copy_subgraph()
        self._teammates_models_state = initial_model_state  # Current model of teammate behavior
        self._evaluate_node_queries_fn = evaluate_query_fn  # Return [(state, score), (state, score), ...] list.
        self._max_branches = max_branching_factor
        self._policy_backup_op = policy_backup_op
        self._heuristic = heuristic  # heuristic used for leaf evaluation (needs to be recalculated with new models)
        self.comm_cost = comm_cost  # Cost per instance of communication

        # References to policy graph
        self._state_graph_map = map_graph(self._policy_root)

        # Helpful references
        self.initial_state_queries = State({})
        #    {Query(name, comm_state): action for name, model in initial_model_state.items()
        #     for comm_state, action in model.previous_communications.items()})
        self._initial_policy_state = State({'Queries': State(self.initial_state_queries), 'End?': False})

        # Caches of commonly computed items
        self._policy_cache = {}
        self._model_state_cache = {self._initial_policy_state: self._teammates_models_state}
        self._policy_graph_cache = {self._initial_policy_state: self._policy_root}
        self._value_of_info = {self._initial_policy_state: 0}

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
                   Query(target_agent_name, node.state['World State']) in policy_state['Queries'] or \
                   len(node.action_space.individual_actions(target_agent_name)) <= 1

        # Need to consider all teammates involved.
        query_evaluations = []
        policy_graph = self._get_policy_graph(policy_state)
        for target_agent in self._teammates_models_state:
            query_evaluations.extend((Query(target_agent, state), value) for state, value in
                                     self._evaluate_node_queries_fn(policy_graph, target_agent, prune_query))

        # Ensure that nlargest actually keeps unique queries, and not multipe of the same query,
        # resulting from different nodes having different evaluations (due to different models)
        action_set = set(Action({self._agent_identity: query_val[0]}) for query_val in
                         nlargest(self._max_branches if self._max_branches != infinity else len(query_evaluations),
                                  query_evaluations, key=lambda qv: qv[1]))

        # Add 'Halt" action for terminating queries
        action_set.add(Action({self._agent_identity: 'Halt'}))
        return action_set

    def transition(self, policy_state, action):
        """
        For a given query, consider the potential reponses from the modeled agent. Then construct a new policy state
        for each possibility.
        """
        print('Transition', policy_state, action)
        # Get the agent's action
        query_action = action[self._agent_identity]

        # Termination action
        if query_action == 'Halt':
            return Distribution({policy_state.update({'End?': True}): 1.0})

        # The transition probabilities are dependent on the model formed by all current queries.
        target_agent_name, query_world_state = query_action
        model_state = self._update_model_state(self._policy_root.state['Models'], policy_state)
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
        assert abs(sum(predicted_responses.values()) - 1.0) < 10e-6, \
            'Predicted responses distribution does not sum to 1: ' + str(abs(sum(predicted_responses.values())))
        assert abs(sum(resulting_states.values()) - 1.0) < 10e-6, \
            'Predicted query action distribution does not sum to 1: ' + str(abs(sum(resulting_states.values())))

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

        print('temp comm scenario end()')
        return False
        # model_state = self._get_policy_graph(policy_state).state['Models']
        model_state = self._update_model_state(self._policy_root.state['Models'], policy_state)
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
        The expected utility for a single query is given as the change in the TOTAL value of information given the
        policy state (as taken from the initial policy). In short, taking query action a gives this much EXTRA value
        for the complete set of new information.

        R(I, a, I') = [V(s_0, π_t | I') - V(s_0, π_0 | I')] - [V(s_0, π_{t-1} | I) - V(s_0, π_0 | I)]
        """
        print('Utility -', action)
        # If the action is to stop communicating, there is no further utility gain.
        if action[self._agent_identity] == 'Halt':
            return 0

        if new_policy_state in self._value_of_info:
            return self._value_of_info[new_policy_state] - self._value_of_info[old_policy_state]

        new_policy_ev = {}
        old_policy_ev = {}

        def compute_policy_ev_update(node, _):
            # leaf check
            if not node.successors:
                new_policy_ev[node] = node.future_value
                old_policy_ev[node] = node.future_value
                return

            new_model_state = self._update_model_state(node.state['Models'], new_policy_state)

            # make new predictions
            new_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                               for other_agent, other_agent_model in new_model_state.items()}

            # Compute original policy action
            old_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                               for other_agent, other_agent_model in node.state['Models'].items()}
            old_action, _ = max(individual_agent_action_values(self._agent_identity, old_predictions,
                                                               node.action_space,
                                                               node.action_values()).items(),
                                key=lambda p: p[1])

            # calc new joint action values
            new_action_values = {
                joint_action: sum(
                    probability * (node.successor_transition_values[(successor.state, joint_action)]
                                   + old_policy_ev[successor])
                    for successor, probability in successor_distribution.items())
                for joint_action, successor_distribution in node.successors.items()}

            # new individual action values
            new_individual_action_values = individual_agent_action_values(self._agent_identity, new_predictions,
                                                                          node.action_space,
                                                                          new_action_values)

            # update policy evs
            old_policy_ev[node] = new_individual_action_values[old_action]
            new_policy_ev[node] = max(new_individual_action_values.values())
            return

        policy_graph = self._policy_root
        depth_map = map_graph_by_depth(policy_graph)

        traverse_graph_topologically(depth_map, node_fn=compute_policy_ev_update, top_down=False)

        # get value of information for new policy update
        new_value_of_info = new_policy_ev[policy_graph] - old_policy_ev[policy_graph]
        self._value_of_info[new_policy_state] = new_value_of_info

        # recall value of information for old policy update
        old_value_of_info = self._value_of_info[old_policy_state]

        return new_value_of_info - old_value_of_info

    def old_utility(self, old_policy_state, action, new_policy_state):
        """
        The expected utility for a single query is given as the change in the TOTAL value of information given the
        policy state (as taken from the initial policy). In short, taking query action a gives this much EXTRA value
        for the complete set of new information.

        R(I, a, I') = [V(s_0, π_t | I') - V(s_0, π_0 | I')] - [V(s_0, π_{t-1} | I) - V(s_0, π_0 | I)]
        """
        print('Utility', action)
        # If the action is to stop communicating, there is no further utility gain.
        if action[self._agent_identity] == 'Halt':
            return 0

        old_policy_new_evs = {}

        def compute_policy_ev_update(nodes, _):
            old_node, new_node = nodes
            if not old_node.successors:
                old_policy_new_evs[new_node] = new_node.future_value
                return

            # First step: compute original policy action
            old_predictions = {other_agent: other_agent_model.predict(old_node.state['World State'])
                               for other_agent, other_agent_model in old_node.state['Models'].items()}
            old_action, _ = max(individual_agent_action_values(self._agent_identity, old_predictions,
                                                               old_node.action_space, old_node.action_values()).items(),
                                key=lambda p: p[1])

            # Need to calculate new action value given the new information
            new_predictions = {other_agent: other_agent_model.predict(new_node.state['World State'])
                               for other_agent, other_agent_model in new_node.state['Models'].items()}

            # DEBUG
            for joint_action, successor_distribution in new_node.successors.items():
                for successor, probability in successor_distribution.items():
                    if (successor.state, joint_action) not in new_node.successor_transition_values:
                        print('Predecessor: ', new_node.state)
                        print('Missing: ', (successor.state, joint_action))
                        for key in new_node.successor_transition_values:
                            print(key, key == (successor.state, joint_action))
                        print('nooo')
            # END DEBUG

            new_action_values = {
                joint_action: sum(
                    probability * (new_node.successor_transition_values[(successor.state, joint_action)]
                                   + old_policy_new_evs[successor])
                    for successor, probability in successor_distribution.items())
                for joint_action, successor_distribution in new_node.successors.items()}

            old_policy_new_evs[new_node] = sum(new_action_values[joint_action] * \
                                               reduce(mul,
                                                      [new_predictions[other_agent][joint_action[other_agent]]
                                                       for other_agent in new_predictions])
                                               for joint_action in
                                               new_node.action_space.fix_actions({self._agent_identity: [old_action]}))

        # Grab old policy and new policy graph
        old_root = self._get_policy_graph(old_policy_state)
        new_root = self._get_policy_graph(new_policy_state, old_policy_state)
        old_depth_map = map_graph_by_depth(old_root)
        new_depth_map = map_graph_by_depth(new_root)
        assert all(old[0].state['World State'] == new[0].state['World State'] for old, new in
                   zip(old_depth_map.items(), new_depth_map.items())), \
            'World states do not align in depth map.'

        comb_depth_map = {(old[0], new[0]): old[1] for old, new in
                          zip(old_depth_map.items(), new_depth_map.items())}

        # evaluate old policy under new policy graph/beliefs
        traverse_graph_topologically(comb_depth_map, node_fn=compute_policy_ev_update, top_down=False)

        return new_root.future_value - old_policy_new_evs[new_root] - self.comm_cost

    def _update_model_state(self, old_model_state, policy_state):
        # Update the model state given the query information
        new_model_state = old_model_state
        for query, response in policy_state['Queries'].items():
            new_model = new_model_state[query.agent].update(query.state, response)
            new_model_state = new_model_state.update_item(query.agent, new_model)  # we can cache this

        return new_model_state

    def _get_policy_graph(self, policy_state):
        """
        Rather than save each bit of information separately (policy, EV, model state, etc), just keep
        a single copy of the policy graph for each policy state. A new policy calculation is needed
        for each step of the communicative search model anyway.
        """
        # Early exit: If we have already calculated this new policy graph, return the cached version.
        if policy_state in self._policy_graph_cache:
            return self._policy_graph_cache[policy_state]

        # Vars for new policy graph
        new_graph = self._policy_root.copy_subgraph()
        depth_map = map_graph_by_depth(new_graph)
        old_to_new_state = {}

        # With new models set, recalculate the policy. This must be done bottom-up.
        def update_graph(node, _):
            # Update the model state given the new query information
            new_model_state = self._update_model_state(node.state['Models'], policy_state)

            old_state = node.state
            node.state = old_state.update_item('Models', new_model_state)
            old_to_new_state[old_state] = node.state

            # Update the node value
            if node.successors:
                # Update transition utils
                new_vals = {}
                for (old_succ, action), value in node.successor_transition_values.items():
                    new_vals[(old_to_new_state[old_succ], action)] = value

                node.successor_transition_values = new_vals

                # Update successors
                for action, successor_distribution in node.successors.items():
                    node.successors[action] = {succ: prob for succ, prob in successor_distribution.items()}

                # predecessors?


                # Update node value, via backup operator
                node._has_changed = True
                self._policy_backup_op(node, self._agent_identity)
            else:
                node.future_value = self._heuristic(node.state)


        import time
        start = time.time()
        print('Pre-update')
        traverse_graph_topologically(depth_map, update_graph, top_down=False)

        mid = time.time()
        print('Traverse: ', mid-start)

        #total_preds = 0
        for node in depth_map:
            #total_preds += len(node.predecessors)
            node.predecessors = set(node.predecessors)
            node._has_changed = False

        print('Second pass', time.time() - mid)
        #print('Total nodes: ', len(depth_map))
        #print('Total predecessors: ', total_preds)

        # Store and return new policy graph.
        self._policy_graph_cache[policy_state] = new_graph

        return new_graph


def communicate(agent, agent_dict, passes, comm_heuristic, branching_factor=infinity, domain_heuristic=lambda s: 0, comm_cost=0):
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
                                 evaluate_query_fn=comm_heuristic,
                                 max_branching_factor=branching_factor,
                                 heuristic=domain_heuristic,
                                 comm_cost=comm_cost)  # backup op?!

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
