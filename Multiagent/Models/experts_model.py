from collections import defaultdict

from MDP.Distribution import Distribution
from Multiagent.Model import Model


class ExpertsModel(Model):

    def __init__(self, scenario, expert_distribution):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            prior - previous counts from before this initialization
            default - the default factory for the default dict
        """
        self.scenario = scenario

        assert isinstance(expert_distribution, dict)
        self.experts = Distribution(expert_distribution)

    def predict(self, state):
        """
        Predicts based on number of prior observations of actions.
        """
        action_distribution = defaultdict(float)
        for expert_predict, expert_prob in self.experts.items():
            for action, action_prob in expert_predict(state).items():
                action_distribution[action] += expert_prob * action_prob

        return action_distribution

    def update(self, state, action):
        """
        Updates the probability according to Bayes' Rule.

        P(expert | action) = P(action | expert) * P(expert) / P(action)
        """
        for expert_predict in self.experts:
            predictions = expert_predict(state)
            self.experts[expert_predict] *= predictions[action]

        self.experts.normalize()

    def __hash__(self):
        return hash(tuple(self.experts.items()))

    def __eq__(self, other):
        pass
