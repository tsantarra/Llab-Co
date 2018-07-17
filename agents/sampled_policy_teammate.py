from collections import deque

from mdp.graph_planner import search
from mdp.distribution import Distribution
from mdp.graph_utilities import map_graph_by_depth, map_graph, traverse_graph_topologically

from functools import reduce
from operator import mul
from math import exp, inf
from random import choice


class SampledTeammateGenerator:

    def __init__(self, scenario, identity, min_graph_iterations=inf):
        self.identity = identity
        self.scenario = scenario
        self.min_graph_iterations = min_graph_iterations
        self.__internal_root, self.__node_order = self.setup_optimal_policy_graph(min_graph_iterations)

        self.policy_state_order = []
        self.all_policy_actions = []

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
        num_individual_policies = reduce(mul, (len(set(action[self.identity]
                                                       for action in node.__optimal_joint_actions))
                                               for node in self.__node_order if node.action_space and node in added))
        print('Number of Optimal Individual Policies: ' + str(num_individual_policies))

    def setup_optimal_policy_graph(self, graph_iterations=inf):
        """
        Creates a complete policy search over the state space of the scenario. Prunes out suboptimal joint actions,
        allowing for sampling from only optimal policies as needed.
        """
        root = search(self.scenario.initial_state(), self.scenario, graph_iterations)

        self._depth_map = map_graph_by_depth(root)
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
        return [choice(node.__optimal_joint_actions)[self.identity] for node in self.__node_order if node.action_space]

    def sample_partial_policy(self):
        """
        Returns a policy only complete for reachable states under the single-agent policy.
        """
        policy = {}
        queue = deque()
        queue.append(self.__internal_root)

        while queue:
            node = queue.popleft()

            if not node.action_space:
                continue

            joint_action = choice(node.__optimal_joint_actions)
            individual_action = joint_action[self.identity]
            policy[node.state] = individual_action

            # Not correct. Need to look at all joint actions with this single-agent action.
            for possible_joint_action in node.action_space.fix_actions({self.identity: individual_action}):
                for successor in node.successors[possible_joint_action]:
                    if successor.state not in policy:
                        queue.append(successor)

        return policy

    def sample_teammate(self):
        policy_dict = dict(zip(self.policy_state_order, self.sample_full_policy()))
        return OfflineSampledPolicyTeammate(self.identity, policy_dict, self.scenario)


class OfflineSampledPolicyTeammate:

    def __init__(self, identity, policy, scenario):
        self.identity = identity
        self.policy = policy
        self.scenario = scenario
        self.__hash = None

    def copy(self):
        return OfflineSampledPolicyTeammate(self.identity, self.policy.copy(), self.scenario)

    def get_action(self, state):
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


class OnlineSampledPolicyTeammate:

    def __init__(self, identity, scenario, rationality=1.0, min_graph_iterations=inf):
        self.identity = identity
        self.scenario = scenario
        self.rationality = rationality
        self.min_graph_iterations = min_graph_iterations

        # Variables for calculating the policy and storing results
        self.internal_root = search(self.scenario.initial_state(), self.scenario)
        self.internal_graph = {}
        self.internal_graph.update(map_graph(self.internal_root))
        self.fixed_policy = {node.state: self.action_probabilities(node).sample()
                             for node in self.internal_graph.values()
                             if node.visits >= self.min_graph_iterations}

    def copy(self):
        new_teammate = OnlineSampledPolicyTeammate(self.identity,
                                                   self.scenario,
                                                   self.rationality,
                                                   self.min_graph_iterations)
        new_teammate.fixed_policy = self.fixed_policy.copy()
        return new_teammate

    def get_action(self, state):
        if state in self.fixed_policy:
            return self.fixed_policy[state]

        return self.predict(state=state).sample()

    def predict(self, state):
        # State not found in partial policy.
        if state not in self.fixed_policy:
            # Perform planning step.
            if state in self.internal_graph:
                node = search(state, self.scenario, self.min_graph_iterations, root_node=self.internal_graph[state])
            else:
                node = search(state, self.scenario, self.min_graph_iterations)

            # Calculate probabilities for actions.
            action_probs = self.action_probabilities(node)
            self.fixed_policy[state] = action_probs.sample()
            actions = action_probs.keys()
        else:  # Already commited to an action.
            actions = self.scenario.actions(state).individual_actions(self.identity)

        return Distribution({action: 1.0 if action == self.fixed_policy[state] else 0
                             for action in actions})

    def action_probabilities(self, node):
        """
        Given an association of values with all joint actions available, return the expectation over each individual
        agent action.
        """
        joint_action_space = node.action_space
        joint_action_values = node.action_values()

        joint_action_probabilities = Distribution({joint_action: exp(self.rationality * value)
                                                   for joint_action, value in joint_action_values.items()})
        joint_action_probabilities.normalize()

        individual_action_probabilities = {action: 0 for action in joint_action_space.individual_actions(self.identity)}
        for joint_action, prob in joint_action_probabilities.items():
            individual_action_probabilities[joint_action[self.identity]] += prob

        assert abs(sum(individual_action_probabilities.values()) - 1) <= 10e-5, 'Probabilities do not add up.'

        return Distribution(individual_action_probabilities)

    def update(self, old_state, observation):
        """ Not recursive. Does not update. May someday wish to update and prune graph. """
        pass

    def __eq__(self, other):
        if self.internal_root.complete and other.internal_root.complete:
            return self.fixed_policy == other.fixed_policy

        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))


if __name__ == '__main__':
    from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario
    from agents.models.chinese_restaurant_process_model import ChineseRestaurantProcessModel

    recipe_scenario = RecipeScenario(num_conditions=3, num_agents=2, num_valid_recipes=1, recipe_size=3)
    generator = SampledTeammateGenerator(recipe_scenario, 'Agent1')
    for recipe in recipe_scenario.recipes:
        print(recipe)

    agg = ChineseRestaurantProcessModel('Agent1', recipe_scenario)

    for i in range(10000):
        pol = generator.sample_full_policy()
        agg.add_teammate_model(pol)

    print(', '.join(str(count) for count in agg.observations.values()))
    print(len(agg.observations))
