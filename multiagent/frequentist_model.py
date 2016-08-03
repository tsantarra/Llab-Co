from collections import defaultdict

from mdp.distribution import Distribution


class FrequentistModel:

    def __init__(self, scenario, prior=None, default_count=1.0):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            prior - previous counts from before this initialization
            default - the default factory for the default dict
        """
        self.scenario = scenario

        if not prior:
            self.counts = defaultdict(lambda: default_count)
        else:
            assert isinstance(prior, dict)
            self.counts = defaultdict(lambda: default_count, prior)

    def predict(self, state):
        """
        Predicts based on number of prior observations of actions.
        """
        actions = self.scenario.actions(state)
        total_observations = sum(self.counts[(state, action)] for action in actions)
        return Distribution({action: self.counts[(state, action)]/total_observations for action in actions})

    def update(self, state, action):
        """
        Updates the count of observations of the action in the given state. Returns a new version!
        """
        new_model = self.copy()
        new_model.counts[(state, action)] += 1
        return new_model

    def copy(self):
        """
        Returns a new copy of this agent model.
        """
        return FrequentistModel(scenario=self.scenario, prior=dict(self.counts))

    def __repr__(self):
        return str(id(self))

    def __eq__(self, other):
        return all(self.counts[key] == other.counts[key] for key in (self.counts.keys() | other.counts.keys()))

    def __hash__(self):
        return hash(tuple(self.counts.items()))