from functools import reduce
from operator import mul

from mdp.graph_planner import search
from mdp.distribution import Distribution
from mdp.graph_utilities import map_graph_by_depth, traverse_graph_topologically

from math import inf
from random import choice
from collections import deque


class SampledTeammateGenerator:

    def __init__(self, scenario, identity, policy_graph=None, min_graph_iterations=inf):
        self.identity = identity
        self.scenario = scenario
        self._flat_policy_graph = []

        if policy_graph:
            self._internal_root = policy_graph
            self._graph_map = {node.state: node for node in map_graph_by_depth(self._internal_root)}
        else:
            self._internal_root, self._graph_map = self.setup_optimal_policy_graph(min_graph_iterations)

        self.policy_stats()

    def policy_stats(self):
        sub_policy_counts = {}

        def count_policies(node, _):
            if not node.action_space:
                sub_policy_counts[node] = 1
                return
            sub_policy_counts[node] = sum(sub_policy_counts[successor]
                                          for joint_action in node._optimal_joint_actions
                                          for successor in node.successors[joint_action])

        map = map_graph_by_depth(self._internal_root)
        traverse_graph_topologically(map, count_policies, top_down=False)

        print('Optimal trajectories: ' + str(sub_policy_counts[self._internal_root]))

    def setup_optimal_policy_graph(self, graph_iterations):
        """
        Creates a complete policy search over the state space of the scenario. Prunes out suboptimal joint actions,
        allowing for sampling from only optimal policies as needed.
        """
        root = search(self.scenario.initial_state(), self.scenario, graph_iterations,
                      heuristic=self.scenario.heuristic if hasattr(self.scenario, 'heuristic') else lambda s: 0)
        depth_map = map_graph_by_depth(root)

        for node in (n for n in depth_map if n.action_space):
            joint_action_values = node.action_values()
            max_action_value = max(joint_action_values.values())
            node._optimal_joint_actions = list(action for action, value in joint_action_values.items()
                                                if abs(value - max_action_value) < 10e-5)

        return root, {node.state: node for node in depth_map}

    def sample_partial_policy(self):
        """
        Returns a policy only complete for reachable states under the single-agent policy.
        """
        policy = {}
        queue = deque()
        queue.append(self._internal_root)

        while queue:
            node = queue.popleft()

            individual_action = choice(node._optimal_joint_actions)[self.identity]
            policy[node.state] = individual_action

            queue.extend(successor for possible_joint_action, successor_dist in node.successors.items()
                         if possible_joint_action[self.identity] == individual_action
                         for successor in successor_dist if successor.action_space and successor.state not in policy)

        return policy

    def sample_teammate(self):
        return SampledPolicyTeammate(self.identity, self.sample_partial_policy(), self.scenario, self)

    def __getstate__(self):
        def _flatten_graph(node, horizon):
            self._flat_policy_graph.append((node.state, node))

        traverse_graph_topologically(map_graph_by_depth(self._internal_root), _flatten_graph)
        return self.__dict__

    def __setstate__(self, container_state):
        self.__dict__.update(container_state)

        node_lookup = dict(self._flat_policy_graph)

        for node in node_lookup.values():
            node.predecessors = {node_lookup[pred_state] for pred_state in node.predecessor_states}
            node._successors = {action: Distribution({node_lookup[state]: prob  for state, prob in node_dist})
                                for action, node_dist in node.flat_successors}
            node._succ_set = set(successor for successor_dist in node.successors.values()
                                 for successor in successor_dist)
            del node.predecessor_states
            del node.flat_successors

        self._flat_policy_graph = []


class SampledPolicyTeammate:

    def __init__(self, identity, policy, scenario, generator):
        self.identity = identity
        self.policy = policy
        self.scenario = scenario
        self._hash = None
        self._generator = generator

    def get_action(self, state):
        if state in self.policy:
            return self.policy[state]

        assert all(key in self._generator._graph_map for key in self._generator._graph_map)

        policy_node = self._generator._graph_map[state]
        if policy_node.action_space:
            return choice(policy_node._optimal_joint_actions)[self.identity]

    def predict(self, state):
        return Distribution({action: 1.0 if action == self.policy[state] else 0
                             for action in self.scenario.actions(state).individual_actions(self.identity)})

    def update(self, old_state, observation):
        pass  # does not update

    def __eq__(self, other):
        if self.__hash__() != other.__hash__():
            return False
        return self.identity == other.identity and self.policy == other.policy

    def __hash__(self):
        if not self._hash:
            self._hash = hash(frozenset(self.policy.items()))

        return self._hash

