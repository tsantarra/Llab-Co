from math import log
from random import uniform


class Distribution(dict):
    """ A distribution of items and their associated probabilities. """

    def __init__(self, args=None):
        """
        Initializes state distribution from list or given distributions.
        """
        if args:
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

    @property
    def entropy(self):
        assert abs(sum(self.values()) - 1.0) < 10e-6, 'Distribution not normalized.'
        return -1 * sum(prob * log(prob) for prob in self.values())

    @property
    def max_item(self):
        assert abs(sum(self.values()) - 1.0) < 10e-6, 'Distribution not normalized.'
        return max(self.items(), key=lambda p: p[1])

    @property
    def max_key(self):
        return self.max_item[0]

    @property
    def max_probability(self):
        return self.max_item[1]

    def copy(self):
        return Distribution(self.items())

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


class ListDistribution:
    """ A distribution of items and their associated probabilities. """

    def __init__(self, args=None):
        """
        Initializes state distribution from list or given distributions.
        """
        if type(args) is list:
            self.__keys, self.__probabilities = zip(*args)
        elif type(args) is dict:
            self.__keys, self.__probabilities = zip(*list(args.items()))

    def expectation(self, value_distribution):
        """
        Returns an expectation over values using the probabilities in this distribution.
        """
        assert len(self) == len(value_distribution),'Improper expectation call: \n{0}\n{1}'.format(self, value_distribution)

        if type(value_distribution) is ListDistribution:
            assert self.__keys == value_distribution.__keys, \
                'Keys do not align between distributions: \n{0}\n{1}'.format(self, value_distribution)
            return sum(prob * value for prob, value in zip(self.__probabilities, value_distribution.values()))

        elif type(value_distribution) is Distribution:
            assert all(key in value_distribution for key in self.__keys), \
                'Keys do not align between distributions: \n{0}\n{1}'.format(self, value_distribution)
            return sum(prob * value_distribution[key] for key, prob in zip(self.__keys, self.__probabilities))

        else:
            raise TypeError('Value distribution has incorrect type ({0}).'.format(type(value_distribution)))

    def conditional_update(self, conditional_probs):
        """
        Given a set of conditional probabilities, the distribution updates itself via Bayes' rule.
        """
        assert len(self) == len(conditional_probs), 'Improper expectation call: \n{0}\n{1}'.format(self, conditional_probs)

        if type(conditional_probs) is ListDistribution:
            assert self.__keys == conditional_probs.__keys, \
                'Keys do not align between distributions: \n{0}\n{1}'.format(self, conditional_probs)
            new_list_dist = ListDistribution([(key, prob * prob2) for key, prob, prob2 in
                                              zip(self.__keys,
                                                  self.__probabilities,
                                                  conditional_probs.values())])

        elif type(conditional_probs) is Distribution:
            assert self.__keys == conditional_probs.__keys, \
                'Keys do not align between distributions: \n{0}\n{1}'.format(self, conditional_probs)

            new_list_dist = ListDistribution([(key, self.__probabilities[index] * conditional_probs[key])
                                              for index, key in enumerate(self.__keys)])
        else:
            raise TypeError('Conditional distribution has incorrect type ({0}).'.format(type(conditional_probs)))

        new_list_dist.normalize()
        return new_list_dist

    def normalize(self):
        """
        Normalizes the distribution such that all probabilities sum to 1.
        """
        total = sum(self.__probabilities)
        assert total > 0, 'Distribution probability total = 0. \n' + str(self)

        self.__probabilities = [prob/total for prob in self.__probabilities]
        return self

    def sample(self):
        """
        Returns a state probabilistically selected from the distribution.
        """
        if not self.__len__():
            raise Exception('Cannot sample from empty distribution.')

        target = uniform(0, sum(self.__probabilities))  # Corrected to sum of probabilities for non-normalized distributions.
        cumulative = 0

        # Accumulate probability until target is reached, returning state.
        for index, probability in enumerate(self.__probabilities):
            cumulative += probability
            if cumulative > target:
                return self.__keys[index]

        # Small rounding errors may cause probability to not reach target for last state.
        return self.__keys[-1]

    def sort(self):
        self.__keys, self.__probabilities = zip(*sorted(self.items()))

    def keys(self):
        return self.__keys

    def values(self):
        return self.__probabilities

    def items(self):
        return zip(self.__keys, self.__probabilities)

    def copy(self):
        return ListDistribution(self.items())

    def __repr__(self):
        return '\nListDistribution {\n' + '\n'.join(str(key) + ' P=' + str(val)
                                                    for key, val in self.items()) + '} /Distribution\n'

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        return self.keys() == other.keys() and self.values() == other.values()

    def __len__(self):
        return len(self.__keys)

    def __key(self):
        return tuple(self.items())

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, other):
        return self.__key() < other.__key()

    def __setitem__(self, item):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.__keys[index], self.__probabilities[index]

    def __iter__(self):
        return self.__keys.__iter__()