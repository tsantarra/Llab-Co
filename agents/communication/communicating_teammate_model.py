from mdp.distribution import Distribution


class CommunicatingTeammateModel:
    """
        This model serves as a wrapper for a purely behavioral model. We explicitely keep track of communicated policy
        information, which overrides the sub-model's predictions in states covered by the queried policy information.
    """

    def __init__(self, teammate_model, scenario, previous_comms=None, previous_obs=None):
        self.model = teammate_model
        self.identity = teammate_model.identity
        self.scenario = scenario

        # Perfect recall of past interaction.
        self.previous_communications = previous_comms if previous_comms else {}
        self.previous_observations = previous_obs if previous_obs else {}

    def predict(self, state):

        if state in self.previous_communications:
            joint_actions = self.scenario.actions(state)
            return Distribution({action: 1.0 if action == self.previous_communications[state] else 0.0
                                 for action in joint_actions.individual_actions(self.identity)})

        if state in self.previous_observations:
            joint_actions = self.scenario.actions(state)
            return Distribution({action: 1.0 if action == self.previous_observations[state] else 0.0
                                 for action in joint_actions.individual_actions(self.identity)})

        return self.model.predict(state)

    def update(self, state, observed_action):
        return CommunicatingTeammateModel(self.model.update(state, observed_action),
                                          self.scenario,
                                          self.previous_communications.copy(),
                                          self.previous_observations.copy())

    def communicated_policy_update(self, state_action_pairs):
        new_comms = self.previous_communications.copy()
        new_comms.update(state_action_pairs)
        return CommunicatingTeammateModel(self.model.batch_update(state_action_pairs),
                                          self.scenario,
                                          new_comms,
                                          self.previous_observations.copy())

    def copy(self):
        return CommunicatingTeammateModel(self.model.copy(),
                                          self.scenario,
                                          self.previous_communications.copy(),
                                          self.previous_observations.copy())

    def __copy__(self):
        return self.copy()

    def __str__(self):
        return 'Communicating Teammate \nModel: {model}\nComms: {comms}'.format(model=str(self.model),
                                                                                comms=len(self.previous_communications))

    def __eq__(self, other):
        return self.previous_communications == other.previous_communications and self.model == other.model

    def __hash__(self):
        return hash(tuple(sorted(self.previous_communications.items()) + [hash(self.model)]))
