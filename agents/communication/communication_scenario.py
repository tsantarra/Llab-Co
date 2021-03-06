"""
The communication process for learning a teammate's intentions in an multiagent MDP can be viewed itself as an MDP,
just one that is exponentially larger than that of the original problem. The idea here is to construct a communication
policy, that is, a series of teammate policy queries that will eventually shape the primary agent's own action policy.
"""
from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import Action
from mdp.graph_planner import search, greedy_action, GraphNode
from mdp.graph_utilities import map_graph_by_depth, traverse_graph_topologically
from agents.modeling_agent import get_max_action, single_agent_policy_backup, individual_agent_action_values
from agents.communication.communicating_teammate_model import CommunicatingTeammateModel

from collections import namedtuple, deque, defaultdict
from math import inf as infinity
from heapq import nlargest
import json
import logging

logger = logging.getLogger()

Query = namedtuple('Query', ['agent', 'state'])


class CommScenario:
    def __init__(self, policy_graph, initial_model_state, identity, evaluate_query_fn,
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
        self._policy_root = policy_graph  # .copy_subgraph()
        self._teammate_names = list(initial_model_state.keys())  # Current model of teammate behavior
        self._evaluate_node_queries_fn = evaluate_query_fn  # Return [(state, score), (state, score), ...] list.
        self._max_branches = max_branching_factor
        self._policy_backup_op = policy_backup_op
        self._comm_cost = comm_cost  # Cost per instance of communication

        # Helpful references
        self._initial_policy_state = State({'Queries': State(((agent, State(model.previous_communications.items()))
                                                              for agent, model in initial_model_state.items())),
                                            'End?': False})

        # Caches of commonly computed items
        self._action_cache = {}
        self._heuristic_value_cache = {}
        self._end_cache = {}
        self._value_of_info = {self._initial_policy_state: 0}

        # Precompute info for first policy state
        self._precompute_info(self._initial_policy_state)

    def initial_state(self):
        """
        The communication state is simply a list of queries and responses along with an 'End?' flag, indicating the
        agent has voluntarily ended the communication process.
        """
        return self._initial_policy_state

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
        assert policy_state in self._action_cache, 'The communicative action set should have been precomputed.'
        return self._action_cache[policy_state]

    def transition(self, policy_state, comm_action):
        """
        For a given query, consider the potential reponses from the modeled agent. Then construct a new policy state
        for each possibility.
        """
        # Get the agent's action
        query_action = comm_action[self._agent_identity]

        # Termination action
        if query_action == 'Halt':
            return Distribution({policy_state.update_item('End?', True): 1.0})

        # The transition probabilities are dependent on the model formed by all current queries.
        target_agent_name, query_world_state = query_action
        model_state = self._update_model_state(self._policy_root.state['Models'], policy_state)
        predicted_responses = model_state[target_agent_name].predict(query_world_state)

        # Consider all possible results
        resulting_states = Distribution([(self._update_policy_queries(policy_state,
                                                                      target_agent_name,
                                                                      query_action.state,
                                                                      action),
                                          probability)
                                         for action, probability in predicted_responses.items()])

        # Sanity check.
        assert abs(sum(predicted_responses.values()) - 1.0) < 10e-6, \
            'Predicted responses distribution does not sum to 1: ' + str(abs(sum(predicted_responses.values())))
        assert abs(sum(resulting_states.values()) - 1.0) < 10e-6, \
            'Predicted query action distribution does not sum to 1: ' + str(abs(sum(resulting_states.values())))

        # Precompute policy state information
        for new_policy_state in resulting_states:
            if new_policy_state not in self._value_of_info:
                self._precompute_info(new_policy_state)

        return resulting_states

    def end(self, policy_state):
        """
        Returns True if given the current policy state, there is no incentive to communicate further.

        Case 1: All reachable states have been queried. (I believe this triggers Case 2, so we may not need an explicit
                check.

        Case 2: We can bound the amount of util to be gained from further communication. If that potential is less
                than the cost of communication, there is no incentive to continue.
                - Bounds: max out of policy util - min in policy util
        """
        if policy_state['End?']:
            return True

        # We have precomputed the end criteria.
        assert policy_state in self._end_cache, 'End criterion should have been precomputed.'
        return self._end_cache[policy_state]

    def utility(self, old_policy_state, action, new_policy_state):
        """
        The expected utility for a single query is given as the change in the TOTAL value of information given the
        policy state (as taken from the initial policy). In short, taking query action a gives this much EXTRA value
        for the complete set of new information.

        R(I, a, I') = [V(s_0, π_t | I') - V(s_0, π_0 | I')] - [V(s_0, π_{t-1} | I) - V(s_0, π_0 | I)]
        """
        # If the action is to stop communicating, there is no further utility gain.
        if action[self._agent_identity] == 'Halt':
            return 0

        # We have already calculated the value of information for this policy state.
        assert new_policy_state in self._value_of_info, 'The value of info should have been precomputed.'
        return self._value_of_info[new_policy_state] - self._value_of_info[old_policy_state] - self._comm_cost

    def _precompute_info(self, policy_state):
        """
        Much of the required MDP information must be gotten from an entirely updated policy graph. Here, we'll
        calculate and store what we need, then delete the copied graph.
        """
        # Generate new graph
        new_policy_root = self._get_policy_graph(policy_state)

        ########################################################################################################
        # Policy EV
        ########################################################################################################
        def compute_policy_ev_update(node, _):
            # leaf check
            if not node.successors:
                node._original_policy_ev = node.future_value
                return

            # New joint action values using original policy values (not equal to the new successor future values)
            new_action_values = {
                joint_action: sum(
                    probability * (node.successor_transition_values[(successor.state, joint_action)]
                                   + successor._original_policy_ev)
                    for successor, probability in successor_distribution.items())
                for joint_action, successor_distribution in node.successors.items()}

            # new individual action values
            new_individual_action_values = individual_agent_action_values(self._agent_identity, node._predictions,
                                                                          node.action_space,
                                                                          new_action_values)

            # Store policy evs
            node._original_policy_ev = new_individual_action_values[node._original_node.optimal_action]
            return

        # Compute EV of old policy given new predictions (in new node graph). Store all info on graph.
        traverse_graph_topologically(new_policy_root._depth_map, compute_policy_ev_update, top_down=False)

        # So currently we calculate Q(s,a) for comms as E[V_{\pi'}(s | I) - V_\pi(s | I)]
        # In truth, this is equal to E[V_{\pi'}(s | I) ] - V_\pi(s) (original EV)
        # We could just store V_{\pi'}(s | I)  and get the original EV off the policy graph in utility()
        new_value_of_info = new_policy_root.future_value - new_policy_root._original_policy_ev
        self._value_of_info[policy_state] = new_value_of_info

        ########################################################################################################
        # End Criterion and Heuristic Evaluation
        ########################################################################################################
        def compute_ev_bounds(node, _):
            # Leaf check
            if not node.successors:
                node._optimistic_ev = node.future_value
                node._pessimistic_ev = node.future_value
                return

            # If communicated policy state, use communicated info. (Fix action)
            world_state = node.state['World State']
            actions_to_fix = {}
            for teammate_name, teammate_model in node.state['Models'].items():
                if world_state in teammate_model.previous_communications:
                    actions_to_fix[teammate_name] = [teammate_model.previous_communications[world_state]]

            # For optimistic EV, choose joint action maximizing expected value;
            optimistic_action_space = node.action_space.fix_actions(actions_to_fix)

            optimistic_action_values = {
                joint_action: sum(
                    probability * (node.successor_transition_values[(successor.state, joint_action)]
                                   + successor._optimistic_ev)
                    for successor, probability in node.successors[joint_action].items())
                for joint_action in optimistic_action_space if joint_action in node.successors}

            # For pessimistic EV, lock in original action and minimize over teammate actions.
            pessimistic_action_space = optimistic_action_space.fix_actions({self._agent_identity:
                                                                                [node._original_node.optimal_action]})
            pessimistic_action_values = {
                joint_action: sum(
                    probability * (node.successor_transition_values[(successor.state, joint_action)]
                                   + successor._pessimistic_ev)
                    for successor, probability in node.successors[joint_action].items())
                for joint_action in pessimistic_action_space if joint_action in node.successors}

            # Store policy evs
            node._optimistic_ev = max(optimistic_action_values.values())
            node._pessimistic_ev = min(pessimistic_action_values.values())

            # The pessimistic ev is no longer guaranteed to be <= the original if we've pruned out possiblities
            # assert (node._optimistic_ev - node._original_policy_ev > -10e-5
            #        and node._original_policy_ev - node._pessimistic_ev > -10e-5), \
            #    "EV bounds calculation incorrect."

            return

        # Compute EV of old policy given new predictions (in new node graph). Store all info on graph.
        traverse_graph_topologically(new_policy_root._depth_map, compute_ev_bounds, top_down=False)

        # The greatest Value of Information with the existing constraints is given by:
        max_value_of_info = new_policy_root._optimistic_ev - new_policy_root._pessimistic_ev

        # The heuristic value, then, is the largest VOI - the current VOI (delta VOI steps) less the cost of at least
        # one query. Of course, if this is less than 0, there is no point in continuing, so the agent should stop.
        self._heuristic_value_cache[policy_state] = max(max_value_of_info - new_value_of_info - self._comm_cost, 0)

        # The end condition: if max possible value of information - cost of all queries it took to get it is less than
        # the cost of a one more comm, then we know we can't afford the cost of further queries.
        is_terminal = max_value_of_info - self._comm_cost * sum(len(queries) for queries in policy_state['Queries'].values()) \
                                        <= self._comm_cost
        self._end_cache[policy_state] = is_terminal

        ########################################################################################################
        # Queries
        ########################################################################################################
        if not is_terminal:
            # Call provided query selector function
            def prune_query(node, target_agent_name):
                return node.scenario_end or \
                       node.state['World State'] in policy_state['Queries'][target_agent_name] or \
                       not node.action_space or \
                       len(node.action_space.individual_actions(target_agent_name)) <= 1

            # Need to consider all teammates involved.
            query_evaluations = []
            for target_agent in self._teammate_names:
                query_evaluations.extend((Query(target_agent, state), value) for state, value in
                                         self._evaluate_node_queries_fn(new_policy_root, new_policy_root._depth_map,
                                                                        target_agent, self._agent_identity, prune_query))

            # Ensure that nlargest actually keeps unique queries, and not multipe of the same query,
            # resulting from different nodes having different evaluations (due to different models)
            action_set = set(Action({self._agent_identity: query_val[0]}) for query_val in
                             nlargest(min(self._max_branches, len(query_evaluations)),
                                      query_evaluations, key=lambda qv: qv[1]))

            # Add 'Halt" action for terminating queries
            action_set.add(Action({self._agent_identity: 'Halt'}))
            self._action_cache[policy_state] = action_set

        ########################################################################################################
        # Cleanup
        ########################################################################################################
        for node in new_policy_root._depth_map.values():
            del node

        del new_policy_root

    def heuristic_value(self, policy_state):
        """
        Heuristic used for communication policy search.

        Note: The admissible heuristic vastly overestimates the potential value of information remaining. In combination
        with partial/online planning, it can cause poor query choices. We will leave choice of a communication heuristic
        as an open direction for research and adopt a flat heuristic (h=0) for the remaining experiments.

        # Old version:
        if policy_state['End?']:
            return 0

        assert policy_state in self._heuristic_value_cache, 'Cannot evaluate heuristic on policy state: ' + policy_state
        return self._heuristic_value_cache[policy_state]
        """
        return 0

    def _update_model_state(self, old_model_state, policy_state):
        assert all(type(model) is CommunicatingTeammateModel for model in old_model_state.values()), \
            'Cannot update non-communicating teammate models with policy queries. '
        # Update the model state given the query information

        queries_by_agent = policy_state['Queries']
        return old_model_state.update(((name, model.communicated_policy_update(queries_by_agent[name].items()))
                                       for name, model in old_model_state.items()))

    def _update_policy_queries(self, policy_state, target_agent_name, state, action):
        # Add query to current queries
        old_agent_query_sets = policy_state['Queries']  # agent: queries (as state/dict)
        new_query_set = old_agent_query_sets[target_agent_name].update_item(state, action)  # update queries (as state)
        new_agent_query_sets = old_agent_query_sets.update_item(target_agent_name, new_query_set)  # update agent: Qs

        return policy_state.update_item('Queries', new_agent_query_sets)  # update policy state

    def _custom_copy_node(self, original_node, predecessor=None):
        # Info we copy
        new_node = GraphNode(original_node.state, original_node.scenario_end, predecessor)
        new_node.complete = original_node.complete
        new_node.action_space = original_node.action_space
        new_node.action_counts = original_node.action_counts
        new_node.optimal_action = original_node.optimal_action
        new_node.visits = original_node.visits
        new_node.future_value = original_node.future_value
        new_node._has_changed = original_node._has_changed
        new_node._old_future_value = original_node._old_future_value
        new_node._action_values = original_node._action_values

        # Info we change
        new_node.successors = {}
        new_node.successor_transition_values = {}
        new_node._incomplete_action_nodes = {}

        # Info we add
        new_node._original_node = original_node
        return new_node

    def _get_policy_graph(self, policy_state):
        """
        Rather than save each bit of information separately (policy, EV, model state, etc), just keep
        a single copy of the policy graph for each policy state. A new policy calculation is needed
        for each step of the communicative search model anyway.
        """
        new_root = self._custom_copy_node(self._policy_root)
        new_root.state = new_root.state.update_item('Models', self._update_model_state(new_root.state['Models'],
                                                                                       policy_state))

        # also need map of graph for dynamic programming things (like pointing to an already generated node!)
        graph_map_by_state = {new_root.state: new_root}

        process_queue = deque([(new_root, 0)])
        while process_queue:
            # have copied node and original
            node, horizon = process_queue.popleft()
            original = node._original_node
            successor_horizon = horizon + 1

            assert original.action_space is not None, 'Leaf node encountered unexpectedly.'

            # reconstruct new action space based on predictions
            node._predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                 for other_agent, other_agent_model in node.state['Models'].items()}
            agent_actions = {agent: set((action for action, prob in predictions.items() if prob > 0.0))
                             for agent, predictions in node._predictions.items()}
            node.action_space = original.action_space.fix_actions(agent_actions)  # JointActionSpace(agent_actions)

            # for only actions in the new action space, consider successors (effectively pruning when possible)
            #   if in graph map, only update predecessors
            #   otherwise, generate new copies, update state and preds, add to queue
            for joint_action in node.action_space:
                if joint_action not in original.successors:
                    continue
                assert joint_action in original.successors, 'Joint action missing while updating policy graph.'

                action_successors = Distribution()
                action_incomplete_succ = set()
                for orig_successor, succ_prob in original.successors[joint_action].items():
                    new_succ_state = orig_successor.state.update_item('Models',
                                                                      self._update_model_state(
                                                                          orig_successor.state['Models'],
                                                                          policy_state))
                    if new_succ_state in graph_map_by_state:
                        # update to existing new successor
                        new_successor = graph_map_by_state[new_succ_state]
                        new_successor.predecessors.add(node)
                    else:
                        # need to make new copy of successor
                        new_successor = self._custom_copy_node(orig_successor, node)
                        new_successor.state = new_succ_state

                        # add to map and process queue
                        graph_map_by_state[new_succ_state] = new_successor
                        if orig_successor.action_space:
                            # check for leaf node, which doesn't need extra processing
                            new_successor._original_node = orig_successor
                            process_queue.append((new_successor, successor_horizon))

                    # Add to node data structures
                    action_successors[new_successor] = succ_prob
                    node.successor_transition_values[(new_succ_state, joint_action)] = \
                        node._original_node.successor_transition_values[(orig_successor.state, joint_action)]
                    if not new_successor.complete:
                        action_incomplete_succ.add(new_successor)

                assert abs(sum(action_successors.values()) - 1.0) < 10e-6, 'Action distribution is not normalized.'
                node.add_new_successors(joint_action, action_successors)
                if action_incomplete_succ:  # do not add if no incomplete successors
                    node._incomplete_action_nodes[joint_action] = action_incomplete_succ

        # Calculate new policy!
        new_root._depth_map = map_graph_by_depth(new_root)
        horizon_lists = defaultdict(list)
        for node, horizon in new_root._depth_map.items():
            horizon_lists[horizon].append(node)

        for node in (n for horizon, nodes_at_horizon in sorted(horizon_lists.items(), reverse=True)
                     for n in nodes_at_horizon):
            # Backup value and flag updates.
            node._old_future_value = node.future_value

            self._policy_backup_op(node, self._agent_identity)

            node._has_changed = (node.future_value != node._old_future_value)
            if not node.complete and node.successors and all(child.complete for child in node.successor_set()):
                node.complete = True
                node._has_changed = True

            # Update node's incomplete actions/child nodes
            for action, child_set in list(node._incomplete_action_nodes.items()):
                child_set = set(child for child in child_set if not child.complete)
                if len(child_set) == 0:
                    del node._incomplete_action_nodes[action]
                else:
                    node._incomplete_action_nodes[action] = child_set

        # Reset changed status.
        for node in new_root._depth_map:
            node._has_changed = False

        return new_root


def communicate(scenario, agent, agent_dict, comm_planning_iterations, comm_heuristic, trial, branching_factor=infinity,
                comm_cost=0):
    """
    This is the primary method for initiating a series of policy queries. It is run as any other scenario:
        - Initiate scenario
        - Plan via graph search
        - Traverse through graph, querying the other agents
        - Recalculate agent models and the primary agent's policy
    """
    if comm_planning_iterations == 0:
        return get_max_action(agent.policy_graph_root, agent.identity), 0

    # Initialize scenario
    comm_scenario = CommScenario(policy_graph=agent.policy_graph_root,
                                 initial_model_state=agent.policy_graph_root.state['Models'],
                                 identity=agent.identity,
                                 evaluate_query_fn=comm_heuristic,
                                 max_branching_factor=branching_factor,
                                 comm_cost=comm_cost)

    original_action = get_max_action(agent.policy_graph_root, agent.identity)
    print('BEGIN COMMUNICATION/// ORIGINAL ACTION: ', original_action)
    print('Current EV: ' + str(agent.policy_graph_root.future_value))
    logger.info('Comm', extra={'EV': agent.policy_graph_root.future_value, 'Action': json.dumps(original_action),
                               'Type': 'Begin', 'Trial': trial})

    current_policy_state = comm_scenario.initial_state()
    comm_graph_node = None
    cost = 0

    while not comm_scenario.end(current_policy_state):
        # Complete graph search
        comm_graph_node = search(state=current_policy_state,
                                 scenario=comm_scenario,
                                 iterations=comm_planning_iterations,
                                 heuristic=comm_scenario.heuristic_value,
                                 root_node=comm_graph_node)

        action = greedy_action(comm_graph_node)
        query_action = action[agent.identity]

        # Initial communication options
        current_policy_state = comm_graph_node.state

        # Check for termination
        if query_action == 'Halt':
            print('Halt')
            break

        # Update cost
        cost += comm_cost

        # Response
        query_target = agent_dict[query_action.agent]
        response = query_target.get_action(query_action.state)
        print('Query:\n{state}\nResponse: {response}'.format(state=query_action.state, response=response))

        logger.info('Query Step', extra={'Query': scenario._serialize_state(query_action.state),
                                         'Response': json.dumps(response), 'Trial': trial})

        # Update position in policy state graph
        current_policy_state = comm_scenario._update_policy_queries(current_policy_state, query_action.agent,
                                                                    query_action.state, response)
        comm_graph_node = comm_graph_node.find_matching_successor(current_policy_state, action=action)

    agent.policy_graph_root = comm_scenario._get_policy_graph(current_policy_state)
    action = get_max_action(agent.policy_graph_root, agent.identity)
    print('END COMMUNICATION/// NEW ACTION:', action)
    print('End EV: ' + str(agent.policy_graph_root.future_value))

    logger.info('Comm', extra={'EV': agent.policy_graph_root.future_value,
                               'New Action': json.dumps(action),
                               'Original Action': json.dumps(original_action),
                               'Type': 'End',
                               'Trial': trial})

    return action, cost
