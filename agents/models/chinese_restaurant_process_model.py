from agents.models.uniform_policy_teammate import UniformPolicyTeammate
from mdp.distribution import ListDistribution

from collections import defaultdict


class ChineseRestaurantProcessModel:

    def __init__(self, identity, scenario, alpha=1):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            prior - previous counts from before this initialization
            default - the default factory for the default dict
        """
        self.scenario = scenario
        self.identity = identity
        self.alpha = alpha
        self.total_obvs = 0
        self.observations = defaultdict(int)

    def add_teammate_model(self, policy_model):
        self.observations[policy_model] += 1
        self.total_obvs += 1

    def prior(self):
        items = [(expert, obvs/(self.total_obvs + self.alpha)) for expert, obvs in self.observations.items()]
        items.append((UniformPolicyTeammate(self.identity, self.scenario), self.alpha / (self.total_obvs + self.alpha)))
        return ListDistribution(items)

