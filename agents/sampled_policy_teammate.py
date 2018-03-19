from mdp.graph_planner import search, map_graph
from mdp.distribution import Distribution

from math import exp, inf
from collections import defaultdict


class SampledPolicyTeammate:

    def __init__(self, identity, scenario, rationality=1.0, min_graph_iterations=inf):
        """ """
        self.identity = identity
        self.scenario = scenario
        self.rationality = rationality
        self.min_graph_iterations = min_graph_iterations

        # Variables for calculating the policy and storing results
        self.internal_root = search(self.scenario.initial_state(), self.scenario)
        self.internal_graph = defaultdict(lambda: None)
        self.internal_graph.update(map_graph(self.internal_root))
        self.fixed_policy = {node.state: self.action_probabilities(node).sample()
                                for node in self.internal_graph.values()
                                if node.visits >= self.min_graph_iterations}

    def copy(self):
        new_teammate = SampledPolicyTeammate(self.identity, self.scenario, self.rationality, self.min_graph_iterations)
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