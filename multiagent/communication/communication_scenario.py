from collections import namedtuple

from mdp.distribution import Distribution
from mdp.scenario import Scenario
from mdp.state import State


def initial_state(policy_graph):
    pass


def queries(policy_state):
    pass


def transition(policy_state, query):
    pass


def end(policy_state):
    pass


def utility(policy_state, query=None):
    pass


grid_scenario = Scenario(initial_state=initial_state, actions=queries,
                         transition=transition, utility=utility, end=end)
