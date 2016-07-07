from collections import Mapping
from copy import copy


class State(Mapping):
    """
    A state is represented as an assignment of state features, as given by the a dictionary of features (keys) and
    their values (feature values). Here, states may be partially specified, creating an abstract state description. As
    such, it is possible to check if a state is contained within another via the __contains__ method ('in').
    """

    def __init__(self, *args, **kwargs):
        """
        Constructs the dict as normal.
        """
        self.__dict = dict(*args, **kwargs)
        self.__hash = None

    def __repr__(self):
        """
        Returns a string representing the state.
        """
        return '\t' + '\n\t'.join(str(key) + ':\t' + str(val) for key, val in sorted(self.items()))

    def copy(self):
        """
        Returns a 'deep-ish' copy of the state instance.
        """
        return self.__class__({key: copy(value) for key, value in self.items()})

    def __copy__(self):
        """
        Returns a copy of the state via the copy method.
        """
        return self.copy()

    def update(self, args):
        """
        Returns a copy with updated state vars, as given by args.
        """
        return State(self.copy(), **args)

    def __hash__(self):
        """
        Returns a hashable form of the state.
        """
        if not self.__hash:
            self.__hash = hash(tuple(self.items()))

        return self.__hash

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __eq__(self, other):
        # print('STATE COMP1:',self.values(), '\t', other.values())
        # print(list((key, type(self[key])) for key in self))
        # print('STATE COMP2:',all(self[key] == other[key] for key in self) and (len(self) == len(other)))

        return (len(self) == len(other)) and all(self[key] == other[key] for key in self)

    def __lt__(self, other):
        return tuple(self.items()) < tuple(other.items())

    def contains_state(self, state):
        """
        Checks state features in self to see if the given state is a specific instance of self. Keys in
        self must be a subset of state's keys.

        Note: should having more keys be 'in' a fewer keys abstraction or vice versa?
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
