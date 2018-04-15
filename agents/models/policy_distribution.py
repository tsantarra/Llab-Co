from mdp.distribution import ListDistribution, Distribution
from agents.sampled_policy_teammate import OfflineSampledPolicyTeammate


class PolicyDistributionModel:

    def __init__(self, scenario, identity, teammate_distribution):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            identity - the name of the agent this object models
            teammate_distribution - a dictionary of teammate models and corresponding probabilities
        """
        self.scenario = scenario
        self.identity = identity
        self.__hash = None

        assert isinstance(teammate_distribution, ListDistribution), \
            'Argument teammate_distribution must be a ListDistribution.'
        self.teammate_distribution = teammate_distribution

    def predict(self, current_state):
        """
        Predicts based on number of prior observations of actions.
        """
        joint_actions = self.scenario.actions(current_state)
        action_distribution = {indiv_action: 0.0 for indiv_action in joint_actions.individual_actions(self.identity)}
        for model, prob in self.teammate_distribution.items():
            if prob < 10e-6:
                continue
            if isinstance(model, OfflineSampledPolicyTeammate):
                action_distribution[model.get_action(current_state)] += prob
            else:
                for act, action_prob in model.predict(current_state).items():
                    action_distribution[act] += action_prob * prob

        return Distribution(action_distribution)

    def update(self, old_state, observed_action):
        """
        Updates the probability according to Bayes' Rule.

        P(teammate | action) = P(action | teammate) * P(teammate) / P(action)
        """
        resulting_probs = []
        for model, prob in self.teammate_distribution.items():
            if isinstance(model, OfflineSampledPolicyTeammate):
                resulting_probs.append((model, prob * float(observed_action == model.get_action(old_state))))
            else:
                prediction = model.predict(old_state)
                resulting_probs.append((model, prob * prediction[observed_action]))

        resulting_model_distribution = ListDistribution(resulting_probs)
        resulting_model_distribution.normalize()
        return PolicyDistributionModel(self.scenario, self.identity, resulting_model_distribution)

    def copy(self):
        """
        Creates a new instance of this model.
        """
        return PolicyDistributionModel(self.scenario, self.identity,
                                       self.teammate_distribution.copy())

    def __str__(self):
        return '\t'.join('TeammateDistributionModel({teammate}) Prob={prob}'.format(teammate=model, prob=prob)
                         for model, prob in self.teammate_distribution.items())

    def __eq__(self, other):
        return self.teammate_distribution == other.teammate_distribution

    def __hash__(self):
        if not self.__hash:
            self.__hash = hash(self.teammate_distribution)

        return self.__hash


if __name__ == '__main__':
    from domains.multi_agent.recipe_sat.recipe_sat_scenario import RecipeScenario
    from agents.models.chinese_restaurant_process_model import ChineseRestaurantProcessModel
    from agents.sampled_policy_teammate import SampledTeammateGenerator

    recipe_scenario = RecipeScenario(num_conditions=3, num_agents=2, num_valid_recipes=1, recipe_size=3)
    generator = SampledTeammateGenerator(recipe_scenario, 'Agent1')
    crp = ChineseRestaurantProcessModel('Agent1', recipe_scenario, alpha=1)

    for i in range(10_000):
        teammate = generator.sample_teammate()
        crp.add_teammate_model(teammate)

    print(', '.join(str(count) for count in crp.observations.values()))

    trial_model = PolicyDistributionModel(recipe_scenario, 'Agent1', crp.prior())

    state = recipe_scenario.initial_state()
    action = trial_model.predict(state).sample()

    print(state)
    print(action)
    print(trial_model)
    new_model = trial_model.update(state, action)
    probs1 = []
    probs2 = []
    for teammate_model in trial_model.teammate_distribution:
        probs1.append(trial_model.teammate_distribution[teammate_model])
        probs2.append((new_model.teammate_distribution[teammate_model]))

    print('\t'.join('{:4.4f}'.format(i) for i in probs1))
    print('\t'.join('{:4.4f}'.format(i) for i in probs2))
