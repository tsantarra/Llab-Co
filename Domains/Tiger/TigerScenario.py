"""
Tiger problem:
    - 2 doors, 1 with a tiger
    - actions: listen, go left, go right
    - observations: hear correct (0.85), hear incorrect (0.15)
    - reward: 1
"""


from MDP.Distribution import Distribution
from MDP.Scenario import Scenario
from MDP.State import State


def initial_state():
    """
    Returns: An initial state for the Coffee Robot Scenario.
    """
    return State({'Tiger': 'Left', 'Player': 'Middle', 'Round': 0})


def transition(state, action):
    new_state = state.copy()

    if action == 'Go left':
        new_state['Player'] = 'Left'
    elif action == 'Go right':
        new_state['Player'] = 'Right'

    new_state['Round'] += 1

    return Distribution({new_state: 1.0})  # all possible outcomes and their associated probabilities


def actions(state):
    """Returns legal actions in the state."""
    if state['Player'] == 'Middle':
        return ['Listen', 'Go left', 'Go right']
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


def utility(state):
    """
    Returns the utility of the state.
    """
    if state['Player'] == 'Middle':
        return 0
    elif state['Player'] == state['Tiger']:
        return -1
    elif state['Player'] != state['Tiger']:
        return 1 * 0.9 ** state['Round']


def end(state):
    """
    Returns True if the scenario has reached an end state.
    """
    return state['Player'] != 'Middle'


tiger_scenario = Scenario(initial_state=initial_state, actions=actions, transition=transition, utility=utility, end=end)