
class PolicyDistributionModel:

    def __init__(self, scenario, identity, policy_distribution, crp_history):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            identity - the name of the agent this object models
            policy_distribution - a distribution over policy indices in crp_history
            crp_history - the Chinese Restaurant Process holding all of the past observed policies
        """
        self.scenario = scenario
        self.identity = identity
        self.policy_distribution = policy_distribution
        self.crp_history = crp_history
        self.__hash = None

    def predict(self, state):
        """
        Predicts based on number of prior observations of actions.
        """
        return self.crp_history.get_action_distribution(state, self.policy_distribution)

    def update(self, state, observed_action):
        """
        Updates the probability according to Bayes' Rule:

            P(teammate | action) = P(action | teammate) * P(teammate) / P(action)
        """
        return PolicyDistributionModel(self.scenario, self.identity,
                                       self.crp_history.posterior(self.policy_distribution, state, observed_action),
                                       self.crp_history)

    def batch_update(self, state_action_pairs):
        """
        Updates the probability according to Bayes' Rule:

            P(teammate | states, actions) = Π[P(action in state | teammate)] * P(teammate) / Z

        where Z is a normalization factor.
        """
        return PolicyDistributionModel(self.scenario, self.identity,
                                       self.crp_history.batch_posterior(self.policy_distribution, state_action_pairs),
                                       self.crp_history)

    def copy(self):
        """
        Creates a new instance of this model.
        """
        return PolicyDistributionModel(self.scenario, self.identity, self.policy_distribution.copy(), self.crp_history)

    def __str__(self):
        return '\t'.join('{teammate}: {prob} '.format(teammate=index, prob=prob)
                         for index, prob in self.policy_distribution)

    def __eq__(self, other):
        if len(self.policy_distribution) != len(other.policy_distribution):
            return False

        return all(index1 == index2 and abs(prob1 - prob2) < 10e-6 for ((index1, prob1), (index2, prob2))
                   in zip(self.policy_distribution, other.policy_distribution))

    def __hash__(self):
        if not self.__hash:
            self.__hash = hash(tuple(index for index, prob in self.policy_distribution))

        return self.__hash

