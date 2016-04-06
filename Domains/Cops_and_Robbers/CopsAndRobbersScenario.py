from MDP.Scenario import Scenario
from MDP.State import State


def initial_state():
    pass


def actions(state):
    pass


def transition(state, action):
    pass


def end(state):
    pass


def utility(state):
    pass


cops_and_robbers_scenario = Scenario(initial_state=initial_state, actions=actions,
                                     transition=transition, utility=utility, end=end)