from agents.models.experts_model import ExpertsModel
from agents.models.uniform_policy_teammate import UniformPolicyTeammate


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

        self.expert_obvs = {}

    def add_observed_policy(self, policy_model):
        if policy_model in self.expert_obvs:
            self.expert_obvs[policy_model] += 1
        else:
            self.expert_obvs[policy_model] = 1

        self.total_obvs += 1

    def get_expert_prior(self):
        expert_dist = {expert: obvs/(self.total_obvs + self.alpha) for expert, obvs in self.expert_obvs.items()}
        expert_dist[UniformPolicyTeammate(self.identity, self.scenario)] = self.alpha / (self.total_obvs + self.alpha)
        return ExpertsModel(self.scenario, expert_dist, self.identity)


