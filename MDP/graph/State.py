import copy

class State(dict):
    def __init__(self, args={}):
        """
        Constructs the dict as normal.
        """
        dict.__init__(self, args)
        self.hash = None

    def __repr__(self):
        """
        Returns a string representing the state.
        """
        s = ''
        for key, val in self.items():
            s += '\n\t' + str(key) + ':\t' + str(val)
        return s

    def __ne__(self, other):
        """
        Checks if other state is not equivalent to this state.
        """
        return not self.__eq__(other)

    def __str__(self):
        """
        Returns a string representing the state.
        """
        return self.__repr__()

    def copy(self):
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
