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

from mdp.distribution import Distribution
from mdp.scenario import Scenario
from mdp.state import State
from mdp.action import JointActionSpace


def initial_state():
    """
    Returns: An initial state for the Coffee Robot Scenario.
    """
    return State({'H': False, 'C': False, 'W': False, 'R': True, 'U': False, 'O': True, 'Round': 0})


def transition(state, action):
    new_state_dict = dict(state.copy())

    act_str = action['Agent']
    if act_str is 'Go':
        # Move to other location.
        new_state_dict['O'] = not new_state_dict['O']

        # Check if the robot gets wet.
        if new_state_dict['R'] and not new_state_dict['U']:
            new_state_dict['W'] = True

    elif act_str is 'BuyCoffee':
        # If in in coffee shop, buy coffee.
        if not new_state_dict['O']:
            new_state_dict['C'] = True

    elif act_str is 'DeliverCoffee':
        # If in office and have coffee, deliver coffee.
        if new_state_dict['O'] and new_state_dict['C']:
            new_state_dict['H'] = True
            new_state_dict['C'] = False

    elif act_str is 'GetUmbrella':
        # If in office, pick up umbrella.
        if new_state_dict['O']:
            new_state_dict['U'] = True

    new_state_dict['Round'] += 1

    return Distribution({State(new_state_dict): 1.0})  # all possible outcomes and their associated probabilities


def actions(state):
    """Returns legal actions in the state."""
    return JointActionSpace({'Agent': ['Go', 'BuyCoffee', 'DeliverCoffee', 'GetUmbrella']})


def utility(old_state, action, new_state):
    """
    Returns the utility of the state.
    """
    util = 0
    if new_state['W']:
        util += -0.5
    if new_state['H']:
        util += 1 * 0.9**(new_state['Round'])

    return util


def end(state):
    """
    Returns True if the scenario has reached an end state.
    """
    return state['H'] or (state['Round'] == 10)


coffee_robot_scenario = Scenario(initial_state=initial_state, actions=actions,
                                 transition=transition, utility=utility, end=end)