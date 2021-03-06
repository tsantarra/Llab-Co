from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace


class CoordinatedActionsScenario:

    def __init__(self, action_set, rounds):
        self.action_set = action_set
        self.rounds = rounds

    def initial_state(self):
        return State({'Seen': (), 'Round': 1})

    def actions(self, state):
        if state['Round'] <= self.rounds:
            return JointActionSpace({'Agent': list(self.action_set), 'Teammate': list(self.action_set)})
        else:
            return JointActionSpace({'Agent': [], 'Teammate': []})

    def transition(self, state, action):
        return Distribution({state.update({'Round': state['Round'] + 1,
                                           'Seen': state['Seen'] + (action,)}): 1.0})

    def end(self, state):
        return state['Round'] > self.rounds

    def utility(self, old_state, action, new_state):
        return 1 if action['Agent'] == action['Teammate'] else 0


class RandomPolicyTeammate:

    def __init__(self, actions, rounds):
        from random import choice
        self.policy = {round: choice(actions) for round in range(1, rounds+2)}
        print('POLICY:', self.policy)

    def get_action(self, state):
        return self.policy[state['Round']]

    def update(self, old_state, observation):
        pass
