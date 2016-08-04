from mdp.distribution import Distribution
from mdp.state import State

class CoordinatedActionsScenario:

    def __init__(self, action_set, rounds):
        self.action_set = action_set
        self.rounds = rounds

    def initial_state(self):
        return State({'Seen': (), 'Round': 1, 'Turn': 'Agent'})

    def actions(self, state):
        return self.action_set

    def transition(self, state, action):
        return Distribution({state.update({'Round': state['Round'] + 1,
                                           'Turn': 'Agent' if state['Turn'] == 'Teammate' else 'Teammate',
                                           'Seen': tuple(state['Seen'] + (action,))}): 1.0})

    def end(self, state):
        return state['Round'] > self.rounds

    def utility(self, old_state, action, new_state):
        return 1 if (new_state['Round'] % 2 == 1 and
                     new_state['Round'] >= 3 and
                     action == new_state['Seen'][-2]) else 0


class SampledPolicyTeammate:

    def __init__(self, actions, rounds):
        from random import choice
        self.policy = {round: choice(actions) for round in range(1, rounds+1)}

    def get_action(self, state):
        return self.policy[state['Round']]

    def update(self, agent_name, old_state, observation, new_state):
        pass