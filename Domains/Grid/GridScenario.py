from collections import namedtuple

from MDP.Distribution import Distribution
from MDP.Scenario import Scenario
from MDP.State import State

GridParams = namedtuple('GridParams', ['x', 'y', 'w', 'h', 'rounds'])
params = GridParams(5, 5, 10, 10, 11)


def initial_state():
    return State({'x': 0, 'y': 0, 'Round': 0})


def actions(state):
        """Returns legal actions in the state."""
        legal_actions = []

        if state['x'] > 0:
            legal_actions += ['left']
        if state['x'] < params.w:
            legal_actions += ['right']
        if state['y'] > 0:
            legal_actions += ['down']
        if state['y'] < params.h:
            legal_actions += ['up']

        return legal_actions


def transition(state, action):
    new_state = state.copy()

    if action is 'right':
        new_state['x'] += 1
    elif action is 'left':
        new_state['x'] += -1
    elif action is 'up':
        new_state['y'] += 1
    elif action is 'down':
        new_state['y'] += -1

    new_state['Round'] += 1

    return Distribution({new_state: 1})  # a list of all possible outcomes and their associated probabilities


def end(state):
    return state['Round'] == params.rounds or (state['x'] == params.x and state['y'] == params.y)


def utility(state, action=None):
    """
    Returns utility associated with given state.
    """
    return int(state['x'] == params.x and state['y'] == params.y)* 0.9**(state['Round'])


grid_scenario = Scenario(initial_state=initial_state, actions=actions,
                         transition=transition, utility=utility, end=end)
