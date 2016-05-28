from mdp.distribution import Distribution
import logging


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
        logging.debug('Model update:')
        logging.debug('State:' + str(state))
        new_model = self.copy()
        for expert_predict in new_model.experts:
            predictions = expert_predict(state)
            logging.debug('Predictions:'+str(predictions))
            new_model.experts[expert_predict] *= predictions[action]

        logging.debug('New vals:' + str(new_model))
        new_model.experts.normalize()
        logging.debug('Normalized:'+str(new_model))
        return new_model

    def copy(self):
        """
        Creates a new instance of this model.
        """
        return ExpertsModel(scenario=self.scenario, expert_distribution=dict(self.experts))

    def __repr__(self):
        return '\t'.join(str(prob) for expert, prob in self.experts.items())

    def __eq__(self, other):
        return self.experts == other.experts

    def __hash__(self):
        return hash(self.experts)

    """ #BAD DOES BAD THINGS.
    def __eq__(self, other):
        return all((key in other.experts) for key in self.experts) and \
               all(self.experts[key] == other.experts[key] for key in self.experts)

    def __hash__(self):
        return hash(tuple(self.experts.items()))
    """
