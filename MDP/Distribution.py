from random import uniform


class Distribution(dict):
    """A distribution of items and their associated probabilities."""

    def __init__(self, args=None):
        """
        Initializes state distribution from list or given distributions.
        """
        if type(args) is list:
            dict.__init__(self, {item: prob for item, prob in args})
        else:
            dict.__init__(self, args)

    def normalize(self):
        """
        Normalizes the distribution such that all probabilities sum to 1.
        """
        total = sum(self.values())

        assert total > 0, "State distribution probability total = 0."

        for item in self.keys():
            self[item] /= total

    def sample(self):
        """
        Returns a state probabilistically selected from the distribution.
        """
        target = uniform(0, 1)
        cumulative = 0

        # Accumulate probability until target is reached, returning state.
        for item, probability in self.items():
            cumulative += probability
            if cumulative > target:
                return item

        # Small rounding errors may cause probability to not reach target for last state.
        return item
