from random import uniform


class Distribution(dict):
    """ A distribution of items and their associated probabilities. """

    def __init__(self, args=None):
        """
        Initializes state distribution from list or given distributions.
        """
        if type(args) is list:
            dict.__init__(self, {item: prob for item, prob in args})
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

        return new_dist.normalize()

    def normalize(self):
        """
        Normalizes the distribution such that all probabilities sum to 1.
        """
        total = sum(self.values())

        assert total > 0, 'State distribution probability total = 0.'

        for item in self.keys():
            self[item] /= total

        return self

    def sample(self):
        """
        Returns a state probabilistically selected from the distribution.
        """
        target = uniform(0, sum(self.values()))  # Corrected to sum of probabilities for non-normalized distributions.
        cumulative = 0

        # Accumulate probability until target is reached, returning state.
        for item, probability in self.items():
            cumulative += probability
            if cumulative > target:
                return item

        # Small rounding errors may cause probability to not reach target for last state.
        return item

    def __repr__(self):
        return '\nDistribution {\n' + '\n'.join(str(key) + '\nP=' + str(val) + '\n'
                                                for key, val in sorted(self.items())) + '} /Distribution\n'

    def __missing__(self, key):
        self[key] = 0.0
        return 0.0

    def __eq__(self, other):
        return all(self[key] == other[key] for key in self) and (len(self) == len(other))

    def __key(self):
        return tuple(self.items())

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, other):
        return self.__key() < other.__key()
