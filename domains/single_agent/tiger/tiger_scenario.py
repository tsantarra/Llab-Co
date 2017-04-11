"""
Tiger problem:
    - 2 doors, 1 with a tiger
    - actions: listen, go left, go right
    - observations: hear correct (0.85), hear incorrect (0.15)
    - reward: 1
"""

from mdp.distribution import Distribution
from mdp.action import JointActionSpace
from mdp.state import State


def initial_state():
    """
    Returns: An initial state for the Coffee Robot Scenario.
    """
    return State({'Tiger': 'Left', 'Player': 'Middle', 'Round': 0})


def transition(state, action):
    new_state_dict = dict(state.copy())

    act_str = action['Agent']
    if act_str == 'Go left':
        new_state_dict['Player'] = 'Left'
    elif act_str == 'Go right':
        new_state_dict['Player'] = 'Right'

    new_state_dict['Round'] += 1

    return Distribution({State(new_state_dict): 1.0})


def actions(state):
    """Returns legal actions in the state."""
    if state['Player'] == 'Middle':
        return JointActionSpace({'Agent': ['Listen', 'Go left', 'Go right']})
    else:
        return []


def observations(state, action):
    """
    Returns probability distributions for observations.
    """
    if state['Tiger'] == 'Left':
        return Distribution({'Roar left': 0.85, 'Roar right': 0.15})
    else:
        return Distribution({'Roar left': 0.15, 'Roar right': 0.85})


def utility(old_state, action, new_state):
    """
    Returns the utility of the state.
    """
    if new_state['Player'] == 'Middle':
        return 0
    elif new_state['Player'] == new_state['Tiger']:
        return -1
    elif new_state['Player'] != new_state['Tiger']:
        return 1 * 0.9 ** new_state['Round']


def end(state):
    """
    Returns True if the scenario has reached an end state.
    """
    return state['Player'] != 'Middle'
