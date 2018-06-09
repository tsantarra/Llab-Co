

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

        for ((index1, prob1), (index2, prob2)) in zip(self.policy_distribution, other.policy_distribution):
            if index1 != index2:
                return False
            if abs(prob1 - prob2) > 10e-6:
                return False

        return True

    def __hash__(self):
        if not self.__hash:
            self.__hash = hash(tuple(index for index, prob in self.policy_distribution))

        return self.__hash


if __name__ == '__main__':
    from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario
    from agents.models.chinese_restaurant_process_model import ChineseRestaurantProcessModel
    from agents.sampled_policy_teammate import SampledTeammateGenerator

    recipe_scenario = RecipeScenario(num_conditions=3, num_agents=2, num_valid_recipes=1, recipe_size=3)
    generator = SampledTeammateGenerator(recipe_scenario, 'Agent1')
    crp = ChineseRestaurantProcessModel('Agent1', recipe_scenario, alpha=1)

    for i in range(10000):
        teammate = generator.sample_teammate()
        crp.add_teammate_model(teammate)

    print(', '.join(str(count) for count in crp.observation_counts.values()))

    trial_model = PolicyDistributionModel(recipe_scenario, 'Agent1', crp.prior())

    state = recipe_scenario.initial_state()
    action = trial_model.predict(state).sample()

    print(state)
    print(action)
    print(trial_model)
    new_model = trial_model.update(state, action)
    probs1 = []
    probs2 = []
    for teammate_model in trial_model.policy_distribution:
        probs1.append(trial_model.policy_distribution[teammate_model])
        probs2.append((new_model.policy_distribution[teammate_model]))

    print('\t'.join('{:4.4f}'.format(i) for i in probs1))
    print('\t'.join('{:4.4f}'.format(i) for i in probs2))
