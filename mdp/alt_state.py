from collections import namedtuple, Mapping


def create_state(name, fields):
    class State(namedtuple(name, fields)):
        # type(var).__name__ gets name

        def __repr__(self):
            """
            Returns a string representing the state.
            """
            return '\t' + '\n\t'.join(str(field) + ':\t' + str(getattr(self, field)) for field in self._fields)

        def __str__(self):
            """
            Returns a string representing the state.
            """
            return self.__repr__()

        def __contains__(self, state):
            """
            Checks state features in self to see if the given state is a specific instance of self. Keys in
            self must be a subset of state's keys.

            Note: should having more keys be 'in' a fewer keys abstraction or vice versa?
            """
            if set(self._fields) - set(state._fields):
                return False
            else:
                return all(getattr(self, field) == getattr(state, field) for field in self._fields)

        def feature_intersection(self, state):
            """
            Returns a new state with features specified only where they are shared between self and state.
            """
            overlapping_features = set(self._fields) & set(state._fields)
            matching_fields = [field for field in overlapping_features if getattr(self, field) == getattr(state, field)]
            new_state_type = create_state(name, matching_fields)
            data = {field: getattr(self, field) for field in matching_fields}
            return new_state_type(**data)

    return State


class FrozenState(dict):
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

    def __setitem__(self, key, value):
        raise AttributeError("can't set attribute")

    def __repr__(self):
        """
        Returns a string representing the state.
        """
        return '\t' + '\n\t'.join(str(key) + ':\t' + str(val) for key, val in self.items())

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
            # flatten out dictionaries
            items = list(self.keys()) + list(self.values())
            hash_list = []
            while items:
                item = items.pop()
                if isinstance(item, dict):
                    for key, val in sorted(item.items()):
                        items.append(key)
                        items.append(val)
                else:
                    hash_list.append(item)

            self.hash = hash(tuple(hash_list))

            # self.hash = hash(tuple(self.items()))
        return self.hash

    def __contains__(self, state):
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


class MappingState(Mapping):
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
        return '\t' + '\n\t'.join(str(key) + ':\t' + str(val) for key, val in self.items())

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
        if self.__hash is None:
            hashes = map(hash, self.items())
            self.__hash = functools.reduce(operator.xor, hashes, 0)

        return self.__hash

    def __contains__(self, state):
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


if __name__ == '__main__':
    Point2d = create_state('Point2d', ['x', 'y'])
    Point3d = create_state('Point3d', ['x', 'y', 'z'])

    a = Point2d(1, 2)
    b = Point3d(4, 2, 3)

    print(a)
    print(b)
    print(b in a)
    print(a in b)
    print(a.feature_intersection(b))
