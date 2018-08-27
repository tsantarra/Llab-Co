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
        return 'State({\t' + '\n\t'.join(str(key) + ':\t' + str(val) for key, val in sorted(self.__dict.items())) + '})'

    def copy(self):
        """
        Returns a 'deep-ish' copy of the state instance.
        """
        return self.__class__({key: copy(value) for key, value in self.__dict.items()})

    def __copy__(self):
        """
        Returns a copy of the state via the copy method. Used in the event someone uses copy(state) from the
        copy module of the standard library.
        """
        return self.copy()

    def update(self, args):
        """
        Returns a copy with updated state vars, as given by args.
        """
        new_copy = self.copy()
        new_copy.__dict.update(args)
        return new_copy

    def update_item(self, key, value):
        """
        Returns a copy with updated item.
        """
        new_copy = self.copy()
        new_copy.__dict[key] = value
        return new_copy

    def remove(self, keys):
        """
        Returns a copy with the specified keys removed.
        """
        return State({key: self.__dict[key] for key in (self.__dict.keys() - set(keys))})

    def __hash__(self):
        """
        Returns a hashable form of the state.
        """
        if not self.__hash:
            self.__hash = hash(tuple(sorted(self.__dict.items())))

        return self.__hash

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __eq__(self, other):
        return self.__dict == other.__dict

    def __lt__(self, other):
        return tuple(sorted(self.__dict.items())) < tuple(sorted(other.__dict.items()))

    def contains_state(self, state):
        """
        Checks state features in self to see if the given state is a specific instance of self. Keys in
        self must be a subset of state's keys.

        Note: should having more keys be 'in' a fewer keys abstraction or vice versa?
        """
        if self.keys() - state.keys():
            return False
        else:
            return all(val == state[key] for key, val in self.__dict.items())

    def feature_intersection(self, state):
        """
        Returns a new state with features specified only where they are shared between self and state.
        """
        return State({key: self[key] for key in (self.keys() & state.keys()) if self[key] == state[key]})


if __name__ == '__main__':
    import timeit

    data = {'x': list(range(5)), 'y': list(range(10))}
    state1 = State(data)
    state2 = State(data)

    t1 = timeit.Timer("state1 == state2", "from __main__ import state1, state2")
    t2 = timeit.Timer("state1.dict_eq(state2)", "from __main__ import state1, state2")
    print('t1', t1.timeit(100000) / 100000)
    print('t2', t2.timeit(100000) / 100000)