from MDP.graph.State import State
from random import uniform


class StateDistribution(dict):
    """A distribution of possible states and their associated probabilities."""

    def __init__(self, args=None):
        """
        Initializes state distribution from list or given distributions.
        """
        if type(args) is State:
            dict.__init__(self, {args:1.})
        elif type(args) is list:
            dict.__init__(self, {state: 1./len(args) for state in args})
        else:
            dict.__init__(self, args)

    def normalize(self):
        """
        Normalizes the distribution such that all probabilities sum to 1.
        """
        total = sum(self.values())

        assert total > 0, "State distribution probability total = 0."

        for state in self.keys():
            self[state] /= total

    def sample(self):
        """
        Returns a state probabilistically selected from the distribution.
        """
        target = uniform(0, 1)
        cumulative = 0

        # Accumulate probability until target is reached, returning state.
        for state, prob in self.items():
            cumulative += prob
            if cumulative > target:
                return state

        # Small rounding errors may cause probability to not reach target for last state.
        return self.keys()[-1]
