from collections import deque

from mdp.graph_planner import search
from mdp.distribution import Distribution
from mdp.graph_utilities import map_graph_by_depth, traverse_graph_topologically

from functools import reduce
from operator import mul
from math import inf
from random import choice


class SampledTeammateGenerator:

    def __init__(self, scenario, identity, min_graph_iterations=inf):
        self.identity = identity
        self.scenario = scenario
        self.min_graph_iterations = min_graph_iterations
        self.__internal_root, self.__node_order = self.setup_optimal_policy_graph(min_graph_iterations)

        self.policy_state_order = []
        self.all_policy_actions = []
        self._depth_map = None
        self._graph_map = {}

        for node in self.__node_order:
            if not node.action_space:
                continue
            self.policy_state_order.append(node.state)
            self.all_policy_actions.append(set(action[identity] for action in node.action_values()))

    def policy_stats(self):
        queue = deque()

        queue.append(self.__internal_root)
        added = {self.__internal_root}

        optimal_policies = 1

        while queue:
            node = queue.popleft()
            if not node.action_space:
                continue

            optimal_policies *= len(node.__optimal_joint_actions)
            for joint_action in node.__optimal_joint_actions:
                for successor in node.successors[joint_action]:
                    if successor not in added:
                        queue.append(successor)
                        added.add(successor)

        sub_policy_counts = {}

        def count_policies(node, horizon):
            if not node.action_space:
                sub_policy_counts[node] = 1
                return

            sub_policy_counts[node] = sum(sub_policy_counts[successor]
                                          for joint_action in node.__optimal_joint_actions
                                          for successor in node.successors[joint_action])

        traverse_graph_topologically(self._depth_map, count_policies, top_down=False)

        print('Optimal trajectories: ' + str(sub_policy_counts[self.__internal_root]))
        print('New Count of Optimal Joint Policies: ' + str(optimal_policies))
        print('Number of Non-Terminal Nodes: ' + str(len(list(node for node in self.__node_order
                                                              if node.action_space and node in added))))
        print('Optimal actions: ' + str(list(len(node.__optimal_joint_actions)
                                             for node in self.__node_order if node.action_space and node in added)))
        print('Number of Optimal Joint Policies: ' + str(reduce(mul, (len(node.__optimal_joint_actions)
                                                                      for node in self.__node_order
                                                                      if node.action_space and node in added))))
        print('Number of Optimal Individual Policies: ' + str(reduce(mul, (len(set(action[self.identity]
                                                                           for action in node.__optimal_joint_actions))
                                                                           for node in self.__node_order
                                                                           if node.action_space and node in added))))

    def setup_optimal_policy_graph(self, graph_iterations=inf):
        """
        Creates a complete policy search over the state space of the scenario. Prunes out suboptimal joint actions,
        allowing for sampling from only optimal policies as needed.
        """
        root = search(self.scenario.initial_state(), self.scenario, graph_iterations)

        self._depth_map = map_graph_by_depth(root)
        self._graph_map = {node.state: node for node in self._depth_map}
        node_list = list((horizon, node) for node, horizon in self._depth_map.items())
        node_list.sort(reverse=True)
        node_order = [node for horizon, node in node_list]

        for node in node_order:
            if not node.action_space:
                continue

            joint_action_values = node.action_values()
            max_action_value = max(joint_action_values.values())
            node.__optimal_joint_actions = list(action for action, value in joint_action_values.items()
                                                if abs(value - max_action_value) < 10e-5)

        return root, node_order

    def sample_full_policy(self):
        """
        Returns a teammate sampled from the stored policy graph.
        """
        return [(node.state, choice(node.__optimal_joint_actions)[self.identity])
                for node in self.__node_order if node.action_space]

    def sample_partial_policy(self):
        """
        Returns a policy only complete for reachable states under the single-agent policy.
        """
        policy = {}
        queue = deque()
        queue.append(self.__internal_root)

        while queue:
            node = queue.popleft()

            individual_action = choice(node.__optimal_joint_actions)[self.identity]
            policy[node.state] = individual_action

            for possible_joint_action in node.action_space:
                if possible_joint_action[self.identity] == individual_action:
                    for successor in node.successors[possible_joint_action]:
                        if successor.action_space and successor.state not in policy:
                            queue.append(successor)

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
        if state not in self.policy:
            return choice(self.__generator._graph_map[state].__optimal_joint_actions)[self.identity]

        return self.policy[state]

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
            self.__hash = hash(self.policy)

        return self.__hash


if __name__ == '__main__':
    from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario
    from agents.models.chinese_restaurant_process_model import SparseChineseRestaurantProcessModel

    recipe_scenario = RecipeScenario(num_conditions=3, num_agents=2, num_valid_recipes=1, recipe_size=3)
    generator = SampledTeammateGenerator(recipe_scenario, 'Agent1')
    for recipe in recipe_scenario.recipes:
        print(recipe)

    agg = SparseChineseRestaurantProcessModel('Agent1', recipe_scenario)

    for i in range(10000):
        pol = generator.sample_full_policy()
        agg.add_teammate_model(pol)

    print(', '.join(str(count) for count in agg.observations.values()))
    print(len(agg.observations))
