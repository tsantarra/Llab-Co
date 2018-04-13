from random import uniform


class Distribution(dict):
    """ A distribution of items and their associated probabilities. """

    def __init__(self, args=None):
        """
        Initializes state distribution from list or given distributions.
        """
        if type(args) is list:
            super(Distribution, self).__init__({item: prob for item, prob in args})
        elif args:
            super(Distribution, self).__init__(args)
        else:
            super(Distribution, self).__init__()

    def expectation(self, values, require_exact_keys=True):
        """
        Returns an expectation over values using the probabilities in this distribution.
        """
        if require_exact_keys:
            assert self.keys() == values.keys(), \
                'Conditional probabilities keys do not map to distribution.\n' + \
                str(set(values.keys())) + ' != ' + str(self.keys())
            return sum(values[key] * self[key] for key in self)
        else:
            return sum(values[key] * self[key] for key in (self.keys() & values.keys()))

    def conditional_update(self, conditional_probs):
        """
        Given a set of conditional probabilities, the distribution updates itself via Bayes' rule.
        """
        assert self.keys() == conditional_probs.keys(), \
            'Conditional probabilities keys do not map to distribution.\n' + \
            str(set(conditional_probs.keys())) + ' != ' + str(self.keys())

        new_dist = self.copy()

        for key in conditional_probs:
            new_dist[key] *= conditional_probs[key]

        new_dist.normalize()
        return new_dist

    def normalize(self):
        """
        Normalizes the distribution such that all probabilities sum to 1.
        """
        total = sum(self.values())
        assert total > 0, 'Distribution probability total = 0. \n' + str(self)

        for item in self.keys():
            self[item] /= total

    def sample(self):
        """
        Returns a state probabilistically selected from the distribution.
        """
        if not self.__len__():
            raise Exception('Cannot sample from empty distribution.')

        target = uniform(0, sum(self.values()))  # Corrected to sum of probabilities for non-normalized distributions.
        cumulative = 0

        # Accumulate probability until target is reached, returning state.
        item = None
        for item, probability in self.items():
            cumulative += probability
            if cumulative > target:
                return item

        # Small rounding errors may cause probability to not reach target for last state.
        return item

    def copy(self):
        return Distribution({**self})

    def __repr__(self):
        return '\nDistribution {\n' + '\n'.join(str(key) + ' P=' + str(val)
                                                for key, val in self.items()) + '} /Distribution\n'

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        return dict.__eq__(self, other)

    def __key(self):
        return tuple(self.items())

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, other):
        return self.__key() < other.__key()
