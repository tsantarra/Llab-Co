from mdp.graph_planner import search
from mdp.distribution import Distribution
from mdp.graph_utilities import map_graph_by_depth, map_graph, traverse_graph_topologically
from mdp.state import State

from math import exp, inf
from random import choice


class SampledTeammateGenerator:

    def __init__(self, scenario, identity, min_graph_iterations=inf):
        self.identity = identity
        self.scenario = scenario
        self.min_graph_iterations = min_graph_iterations
        self.__internal_root = search(self.scenario.initial_state(), self.scenario, min_graph_iterations)

        node_list = list((horizon, node) for node, horizon in map_graph_by_depth(self.__internal_root).items())
        node_list.sort(reverse=True)
        self.__node_order = [node for horizon, node in node_list]

        self.policy_state_order = []
        self.all_policy_actions = []

        for node in self.__node_order:
            if not node.action_space:
                continue
            self.policy_state_order.append(node.state)
            self.all_policy_actions.append(set(action[identity] for action in node.action_values()))

    def sample_policy(self):
        """
        Returns a teammate sampled from the stored policy graph.
        Process:
            1. Map the graph by depth.
            2. Traverse the graph bottom-up. (Policy choices typically depend on future actions.)
            3. At each node, choose an action.
            4. Construct and return teammate.
        """
        policy = []

        for node in self.__node_order:
            if not node.action_space:
                continue

            joint_action_values = node.action_values()
            max_action_value = max(joint_action_values.values())
            actions = list((index, action_value[0]) for index, action_value in enumerate(joint_action_values.items())
                           if abs(action_value[1]-max_action_value) < 10e-5)
            pick_index, action = choice(actions)
            policy.append(action[self.identity])

        return policy

    def sample_teammate(self):
        policy_dict = dict(zip(self.policy_state_order, self.sample_policy()))
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
        pol = generator.sample_policy()
        agg.add_teammate_model(pol)

    print(', '.join(str(count) for count in agg.observations.values()))
    print(len(agg.observations))
