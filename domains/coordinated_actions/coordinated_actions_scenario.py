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
        return Distribution({state.update({'Round': state['Round'] + 1 if state['Turn'] == 'Teammate' else state['Round'],
                                           'Turn': 'Agent' if state['Turn'] == 'Teammate' else 'Teammate',
                                           'Seen': tuple(state['Seen'] + (action,))}): 1.0})

    def end(self, state):
        return state['Round'] > self.rounds or (state['Round'] >= 2
                                                and state['Turn'] == 'Agent'
                                                and state['Seen'][-1] != state['Seen'][-2])

    def utility(self, old_state, action, new_state):
        return 1 if (new_state['Round'] >= 1 and
                     new_state['Turn'] == 'Agent' and
                     new_state['Seen'][-2] == action) else 0


class SampledPolicyTeammate:

    def __init__(self, actions, rounds):
        from random import choice
        self.policy = {round: choice(actions) for round in range(1, rounds+1)}
        print('POLICY:', self.policy)

    def get_action(self, state):
        return self.policy[state['Round']]

    def update(self, agent_name, old_state, observation, new_state):
        pass