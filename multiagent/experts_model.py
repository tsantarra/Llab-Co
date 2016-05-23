from collections import defaultdict

from mdp.distribution import Distribution


class ExpertsModel:

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
        action_distribution = {action: 0.0 for action in self.scenario.actions(state)}
        for expert_predict, expert_prob in self.experts.items():
            for action, action_prob in expert_predict(state).items():
                action_distribution[action] += expert_prob * action_prob

        return Distribution(action_distribution)

    def update(self, state, action):
        """
        Updates the probability according to Bayes' Rule.

        P(expert | action) = P(action | expert) * P(expert) / P(action)
        """
        new_model = self.copy()
        for expert_predict in new_model.experts:
            predictions = expert_predict(state)
            new_model.experts[expert_predict] *= predictions[action]

        new_model.experts.normalize()
        return new_model

    def copy(self):
        """
        Creates a new instance of this model.
        """
        return ExpertsModel(scenario=self.scenario, expert_distribution=dict(self.experts))

    def __repr__(self):
        return '\t'.join(str(prob) for expert, prob in self.experts.items())
