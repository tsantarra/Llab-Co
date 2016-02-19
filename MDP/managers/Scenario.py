from collections import namedtuple

Scenario = namedtuple('Scenario', ['initial_state', 'transition', 'actions', 'utility', 'end'])
