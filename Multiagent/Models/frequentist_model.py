from collections import defaultdict
from Multiagent.Model import Model
from MDP.Distribution import Distribution


class FrequentistModel(Model):

    def __init__(self, scenario, prior=None, default=float):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            prior - previous counts from before this initialization
            default - the default factory for the default dict
        """
        self.scenario = scenario

        if not prior:
            self.counts = defaultdict(default)
        else:
            assert isinstance(prior, dict)
            self.counts = defaultdict(default, prior)

    def predict(self, state):
        """
        Predicts based on number of prior observations of actions.
        """
        actions = self.scenario.actions(state)
        total_observations = sum(self.counts[(state, action)] for action in actions)
        return Distribution({action: self.counts[(state, action)]/total_observations for action in actions})

    def update(self, state, action):
        """
        Updates the count of observations of the action in the given state.
        """
        self.counts[(state, action)] += 1
