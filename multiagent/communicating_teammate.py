from mdp.distribution import Distribution

class CommunicatingTeammate:

    def __init__(self, teammate_model, previous_comms=None):
        if previous_comms:
            self.previous_communications = previous_comms
        else:
            self.previous_communications = {}

        self.model = teammate_model

    def predict(self, state):
        if state in self.previous_communications:
            return Distribution({self.previous_communications[state]: 1.0})

        return self.model.predict(state)

    def update(self, state, action):
        self.model.update(state, action)

    def communicated_policy_update(self, state_action_pairs):
        self.previous_communications.update(state_action_pairs)

    def copy(self):
        return CommunicatingTeammate(self.model.copy(), self.previous_communications)

    def __repr__(self):
        return str(id(self))

    def __eq__(self, other):
        return self.previous_communications == other.previous_communications and self.model == other.model

    def __hash__(self):
        return hash(tuple(list(self.previous_communications.items()) + [hash(self.model)]))
