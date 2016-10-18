from mdp.distribution import Distribution


class CommunicatingTeammate:

    def __init__(self, teammate_model, scenario, previous_comms=None):
        if previous_comms:
            self.previous_communications = previous_comms
        else:
            self.previous_communications = {}

        self.model = teammate_model
        self.scenario = scenario

    def predict(self, state):

        if state in self.previous_communications:
            actions = self.scenario.actions(state)
            return Distribution({action: 1.0 if action == self.previous_communications[state] else 0.0 for action in actions})

        return self.model.predict(state)

    def update(self, old_state, action):
        return CommunicatingTeammate(self.model.update(old_state, action), self.scenario, self.previous_communications.copy())

    def communicated_policy_update(self, state_action_pairs):
        new_comms = self.previous_communications.copy()
        new_comms.update(state_action_pairs)
        return CommunicatingTeammate(self.model.copy(), self.scenario, new_comms)

    def copy(self):
        return CommunicatingTeammate(self.model.copy(), self.scenario, self.previous_communications.copy())

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        return str(id(self))

    def __eq__(self, other):
        return self.previous_communications == other.previous_communications and self.model == other.model

    def __hash__(self):
        return hash(tuple(list(self.previous_communications.items()) + [hash(self.model)]))
