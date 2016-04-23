"""
For simplicity, we will treat a Scenario as a collection of functions which operate on States.
"""
from collections import namedtuple

Scenario = namedtuple('Scenario', ['initial_state', 'transition', 'actions', 'utility', 'end'])
Scenario.__new__.__defaults__ = (None,) * len(Scenario._fields)  # Sets default values to None

