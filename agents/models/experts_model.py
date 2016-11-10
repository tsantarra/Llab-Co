from mdp.distribution import Distribution
import logging


class ExpertsModel:

    def __init__(self, scenario, expert_distribution, identity):
        """
        Initializes the frequentist model.
            scenario - the scenario for the planner
            prior - previous counts from before this initialization
            default - the default factory for the default dict
        """
        self.scenario = scenario
        self.identity = identity

        assert isinstance(expert_distribution, dict)
        self.experts = Distribution(expert_distribution)

    def predict(self, state):
        """
        Predicts based on number of prior observations of actions.
        """
        joint_actions = self.scenario.actions(state)
        action_distribution = {action: 0.0 for action in joint_actions.individual_actions(self.identity)}
        for expert, expert_prob in self.experts.items():
            for action, action_prob in expert.predict(state).items():
                action_distribution[action] += expert_prob * action_prob

        return Distribution(action_distribution)

    def update(self, old_state, action):
        """
        Updates the probability according to Bayes' Rule.

        P(expert | action) = P(action | expert) * P(expert) / P(action)
        """
        logging.debug('Model update:')
        logging.debug('State:' + str(old_state))

        new_model = self.copy()
        for expert in new_model.experts:
            predictions = expert.predict(old_state)
            logging.debug('Predictions:'+str(predictions))
            new_model.experts[expert] *= predictions[action] * 0.9 + (1.0/len(new_model.experts)) * 0.1

        logging.debug('New vals:' + str(new_model))
        new_model.experts.normalize()
        logging.debug('Normalized:'+str(new_model))

        return new_model

    def copy(self):
        """
        Creates a new instance of this model.
        """
        return ExpertsModel(scenario=self.scenario,
                            expert_distribution={key: prob for key, prob in self.experts.items()},
                            identity=self.identity)

    def __str__(self):
        return '\t'.join('Expert({expert}) Prob={prob}'.format(expert=expert, prob=prob)
                         for expert, prob in self.experts.items())

    def __eq__(self, other):
        return all(key in other.experts for key in self.experts) \
                and all(abs(self.experts[key] - other.experts[key]) < 10e-6 for key in self.experts)

    def __hash__(self):
        return hash(self.experts)

