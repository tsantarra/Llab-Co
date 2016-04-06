"""
Coffee robot domain:
    H - User has coffee.
    C - Have coffee.
    W - Wet.
    R - Is raining.
    U - Have umbrella.
    O - In office. If false, the robot is at the coffee shop.
    Round - how many turns have elapsed
"""

from MDP.Distribution import Distribution
from MDP.Scenario import Scenario
from MDP.State import State


def initial_state():
    """
    Returns: An initial state for the Coffee Robot Scenario.
    """
    return State({'H': False, 'C': False, 'W': False, 'R': True, 'U': False, 'O': True, 'Round': 0})


def transition(state, action):
    new_state = state.copy()

    if action is 'Go':
        # Move to other location.
        new_state['O'] = not new_state['O']

        # Check if the robot gets wet.
        if new_state['R'] and not new_state['U']:
            new_state['W'] = True

    elif action is 'BuyCoffee':
        # If in in coffee shop, buy coffee.
        if not new_state['O']:
            new_state['C'] = True

    elif action is 'DeliverCoffee':
        # If in office and have coffee, deliver coffee.
        if new_state['O'] and new_state['C']:
            new_state['H'] = True
            new_state['C'] = False

    elif action is 'GetUmbrella':
        # If in office, pick up umbrella.
        if new_state['O']:
            new_state['U'] = True

    new_state['Round'] += 1

    return Distribution({new_state: 1.0})  # all possible outcomes and their associated probabilities


def actions(state):
    """Returns legal actions in the state."""
    return ['Go', 'BuyCoffee', 'DeliverCoffee', 'GetUmbrella']


def utility(state, action=None):
    """
    Returns the utility of the state.
    """
    util = 0
    if state['W']:
        util += -0.5
    if state['H']:
        util += 1 * 0.9**(state['Round'])

    return util


def end(state):
    """
    Returns True if the scenario has reached an end state.
    """
    return state['H'] or (state['Round'] == 10)


coffee_robot_scenario = Scenario(initial_state=initial_state, actions=actions,
                                 transition=transition, utility=utility, end=end)