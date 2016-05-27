from collections import namedtuple, Mapping
import functools
import operator
from copy import copy, deepcopy


def create_state(name, fields):
    class State(namedtuple(name, fields)):
        # type(var).__name__ gets name

        def copy(self):
            return State(*self)

        def deepcopy(self):
            return self.__class__(*(copy(getattr(self, field)) for field in self._fields))

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
        super(FrozenState, self).__init__(args)
        self.hash = None

    def __setitem__(self, key, value):
        raise AttributeError("can't set attribute")

    def copy(self):
        """
        Returns a copy of the state instance.
        """
        return FrozenState(self)

    def deepcopy(self):
        """
        Returns a copy of the state instance.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy(v))
        return result


class MappingState(Mapping):
    """
    A state is represented as an assignment of state features, as given by the a dictionary of features (keys) and
    their values (feature values). Here, states may be partially specified, creating an abstract state description. As
    such, it is possible to check if a state is contained within another via the __contains__ method ('in').
    """

    def __init__(self, args):
        """
        Constructs the dict as normal.
        """
        self.__dict = dict(**args)
        self.__hash = None

    def copy(self):
        """
        Returns a copy of the state instance.
        """
        return MappingState(self)

    def deepcopy(self):
        """
        Returns a copy of the state instance.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            setattr(result, k, copy(v))
        return result

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __hash__(self):
        if self.__hash is None:
            hashes = map(hash, self.__dict.items())
            self.__hash = functools.reduce(operator.xor, hashes, 0)

        return self.__hash


if __name__ == '__main__':
    import timeit
    from copy import copy, deepcopy
    from mdp.state import State

    state_type = create_state('State', ['x', 'y'])

    data = {'x': list(range(5)), 'y': list(range(10))}
    type1 = state_type(**data)
    type2 = FrozenState(data)
    type3 = MappingState(data)
    type4 = State(data)

    print('\ncopy times')
    t1 = timeit.Timer("o2 = type1.copy()", "from __main__ import type1")
    t2 = timeit.Timer("o2 = type2.copy()", "from __main__ import type2")
    t3 = timeit.Timer("o2 = type3.copy()", "from __main__ import type3")
    t4 = timeit.Timer("o2 = type4.copy()", "from __main__ import type4")
    print('t1',t1.timeit(10000) / 10000)
    print('t2',t2.timeit(10000) / 10000)
    print('t3',t3.timeit(10000) / 10000)
    print('t4',t4.timeit(10000) / 10000)

    print('\ncopy() times')
    t1 = timeit.Timer("o2 = copy(type1)", "from __main__ import type1; from copy import copy, deepcopy")
    #t2 = timeit.Timer("o2 = copy(type2)", "from __main__ import type2; from copy import copy, deepcopy")
    t3 = timeit.Timer("o2 = copy(type3)", "from __main__ import type3; from copy import copy, deepcopy")
    t4 = timeit.Timer("o2 = copy(type4)", "from __main__ import type4; from copy import copy, deepcopy")
    print('t1',t1.timeit(10000) / 10000)
    #print('t2',t2.timeit(10000) / 10000)
    print('t3',t3.timeit(10000) / 10000)
    print('t4',t4.timeit(10000) / 10000)

    print('\ndeepcopy() times')
    t1 = timeit.Timer("o2 = deepcopy(type1)", "from __main__ import type1; from copy import copy, deepcopy")
    #t2 = timeit.Timer("o2 = deepcopy(type2)", "from __main__ import type2; from copy import copy, deepcopy")
    t3 = timeit.Timer("o2 = deepcopy(type3)", "from __main__ import type3; from copy import copy, deepcopy")
    t4 = timeit.Timer("o2 = deepcopy(type4)", "from __main__ import type4; from copy import copy, deepcopy")
    print('t1',t1.timeit(10000) / 10000)
    #print('t2',t2.timeit(10000) / 10000)
    print('t3',t3.timeit(10000) / 10000)
    print('t4',t4.timeit(10000) / 10000)

    print('\ncustom deepcopy times')
    t1 = timeit.Timer("o2 = type1.deepcopy()", "from __main__ import type1")
    t2 = timeit.Timer("o2 = type2.deepcopy()", "from __main__ import type2")
    t3 = timeit.Timer("o2 = type3.deepcopy()", "from __main__ import type3")
    t4 = timeit.Timer("o2 = type4.deepcopy()", "from __main__ import type4")
    print('t1',t1.timeit(10000) / 10000)
    print('t2',t2.timeit(10000) / 10000)
    print('t3',t3.timeit(10000) / 10000)
    print('t4',t4.timeit(10000) / 10000)

