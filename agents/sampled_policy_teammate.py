from collections import deque

from mdp.graph_planner import search
from mdp.distribution import Distribution
from mdp.graph_utilities import map_graph_by_depth

from math import inf
from random import choice


class SampledTeammateGenerator:

    def __init__(self, scenario, identity, min_graph_iterations=inf):
        self.identity = identity
        self.scenario = scenario
        self._internal_root, self._depth_map, self._graph_map = \
            self.setup_optimal_policy_graph(min_graph_iterations)

    def setup_optimal_policy_graph(self, graph_iterations=inf):
        """
        Creates a complete policy search over the state space of the scenario. Prunes out suboptimal joint actions,
        allowing for sampling from only optimal policies as needed.
        """
        root = search(self.scenario.initial_state(), self.scenario, graph_iterations)
        depth_map = map_graph_by_depth(root)

        for node in (n for n in depth_map if n.action_space):
            joint_action_values = node.action_values()
            max_action_value = max(joint_action_values.values())
            node._optimal_joint_actions = list(action for action, value in joint_action_values.items()
                                                if abs(value - max_action_value) < 10e-5)

        return root, depth_map, {node.state: node for node in depth_map}

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

            queue.extend(successor for possible_joint_action
                         in node.action_space  # .fix_actions({self.identity: individual_action})
                         if possible_joint_action[self.identity] == individual_action
                         for successor in node.successors[possible_joint_action]
                         if successor.action_space and successor.state not in policy)

        return policy

    def sample_teammate(self):
        return SampledPolicyTeammate(self.identity, self.sample_partial_policy(), self.scenario, self)


class SampledPolicyTeammate:

    def __init__(self, identity, policy, scenario, generator):
        self.identity = identity
        self.policy = policy
        self.scenario = scenario
        self.__hash = None
        self.__generator = generator

    def get_action(self, state):
        if state in self.policy:
            return self.policy[state]

        policy_node = self.__generator._graph_map[state]
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
        if not self.__hash:
            self.__hash = hash(frozenset(self.policy.items()))

        return self.__hash


if __name__ == '__main__':
    print(inf)