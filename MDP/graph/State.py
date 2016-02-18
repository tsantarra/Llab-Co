from collections import OrderedDict

class State(OrderedDict):
    """
    A state is represented as an assignment of state features, as given by the a dictionary of features (keys) and
    their values (feature values). Here, states may be partially specified, creating an abstract state description. As
    such, it is possible to check if a state is contained within another via the __contains__ method ('in').
    """

    def __init__(self, args={}):
        """
        Constructs the dict as normal.
        """
        super(State, self).__init__(args)
        self.hash = None

    def __repr__(self):
        """
        Returns a string representing the state.
        """
        s = ''
        for key, val in self.items():
            s += '\n\t' + str(key) + ':\t' + str(val)
        return s

    def __str__(self):
        """
        Returns a string representing the state.
        """
        return self.__repr__()

    def __copy__(self):
        """
        Returns a copy of the state instance.
        """
        return State(self)
    
    def __hash__(self):
        """
        Returns a hashable form of the state.
        """
        if not self.hash:
            self.hash = hash(tuple(self.items()))
        return self.hash

    def __contains__(self, state):
        """
        Checks state features in self to see if the given state is a specific instance of self. Keys in
        self must be a subset of state's keys.
        """
        if self.keys() - state.keys():
            return False
        else:
            return all(val == state[key] for key, val in self.items())

    def feature_intersection(self, state):
        """
        Returns a new state with features specified only where they are shared between self and state.
        """
        return State({key: self[key] for key in (self.keys() & state.keys()) if self[key] == state[key]})
