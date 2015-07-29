

class AbstractState(dict):
    def __init__(self, args={}):
        """
        Constructs the dict as normal.
        """
        dict.__init__(self, args)

    def __setitem__(self, key, value):
        """
        Sets the key/val pair in dict, but with val in a set.
        """
        if type(value) is set:
            return dict.__setitem__(self, key, value)
        else:
            return dict.__setitem__(self, key, {value})

    def __getitem__(self, key):
        """
        As __getitem__ in dict, but returns an empty set if the key is not 
        in the state.
        """
        if key in self:
            return dict.__getitem__(self, key)
        else:
            return set()

    def union_state(self, s):
        """
        Returns a new state where the constraints on the state variables are 
        the union of the contraints from the states merged.
        """
        newState = AbstractState()
        keys = set(self.keys()).union(set(s.keys()))
        for key in keys:
            newState[key] = self[key].union(s[key])
        
        return newState

    def intersect_state(self, s):
        """
        Returns a new state where the constraints on the state variables are 
        the intersection of the contraints from the states merged.
        """
        newState = AbstractState()
        keys = set(self.keys()).union(set(s.keys()))
        for key in keys:
            newState[key] = self[key].intersection(s[key])
        
        return newState

    def __repr__(self):
        """
        Returns a string representing the state.
        """
        s = ''
        for key, val in self.items():
            s += '\n\t' + str(key) + ':\t' + str(val)
        return s

    def copy(self):
        """
        Returns a copy of the state.
        """
        return AbstractState(self)