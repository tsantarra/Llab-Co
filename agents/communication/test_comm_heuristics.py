from collections import namedtuple
import unittest

from agents.communication.communication_strategies import *
from mdp.graph_planner import GraphNode
from mdp.action import JointActionSpace, Action
from mdp.state import State


agents = ['Me', 'Other1', 'Other2']
actions = {agent: list(range(3)) for agent in agents}
state = State({'World State': State({}), 'Models': State({})})

MockModel = namedtuple('MockModel', ['predict'])
mock_other1 = MockModel(lambda s: Distribution({0: 0.5, 1: 0.3, 2: 0.2}))
mock_other2 = MockModel(lambda s: Distribution({0: 0.99, 1: 0.01, 2: 0.0}))

action_values = {
    Action({'Other1': 0, 'Other2': 0, 'Me': 0}): 3,  # p=0.495
    Action({'Other1': 0, 'Other2': 0, 'Me': 1}): 2,
    Action({'Other1': 0, 'Other2': 0, 'Me': 2}): 1,

    Action({'Other1': 1, 'Other2': 0, 'Me': 0}): 1,  # p=0.297
    Action({'Other1': 1, 'Other2': 0, 'Me': 1}): 3,
    Action({'Other1': 1, 'Other2': 0, 'Me': 2}): 2,

    Action({'Other1': 2, 'Other2': 0, 'Me': 0}): 1,  # p=0.198
    Action({'Other1': 2, 'Other2': 0, 'Me': 1}): 2,
    Action({'Other1': 2, 'Other2': 0, 'Me': 2}): 3,

    Action({'Other1': 0, 'Other2': 1, 'Me': 0}): 0,  # very unlikely as Other2-1 is p=0.01
    Action({'Other1': 0, 'Other2': 1, 'Me': 1}): 0,
    Action({'Other1': 0, 'Other2': 1, 'Me': 2}): 3,
    Action({'Other1': 1, 'Other2': 1, 'Me': 0}): 0,
    Action({'Other1': 1, 'Other2': 1, 'Me': 1}): 0,
    Action({'Other1': 1, 'Other2': 1, 'Me': 2}): 3,
    Action({'Other1': 2, 'Other2': 1, 'Me': 0}): 0,
    Action({'Other1': 2, 'Other2': 1, 'Me': 1}): 0,
    Action({'Other1': 2, 'Other2': 1, 'Me': 2}): 3,

    Action({'Other1': 0, 'Other2': 2, 'Me': 0}): 0,  # impossible given Other2-2 is p=0
    Action({'Other1': 0, 'Other2': 2, 'Me': 1}): 100,
    Action({'Other1': 0, 'Other2': 2, 'Me': 2}): 0,
    Action({'Other1': 1, 'Other2': 2, 'Me': 0}): 0,
    Action({'Other1': 1, 'Other2': 2, 'Me': 1}): 100,
    Action({'Other1': 1, 'Other2': 2, 'Me': 2}): 0,
    Action({'Other1': 2, 'Other2': 2, 'Me': 0}): 0,
    Action({'Other1': 2, 'Other2': 2, 'Me': 1}): 100,
    Action({'Other1': 2, 'Other2': 2, 'Me': 2}): 0,
}


class MockScenario:
    def end(self):
        return False

    def actions(self):
        return JointActionSpace(actions)


class TestHeuristicFunctions(unittest.TestCase):

    def setUp(self):
        """ Set up test node. """
        node = GraphNode(state, MockScenario())
        node.__action_values = action_values

    def test_local_inf(self):
        pass

    def test_local_val_of_info(self):
        pass

    def test_local_abs_error(self):
        pass

    def test_local_util_var(self):
        pass

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()