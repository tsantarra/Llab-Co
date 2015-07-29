__author__ = 'Trevor'

from MDP.graph.AbstractState import AbstractState
from MDP.managers.StateTransitionManager import StateTransitionManager

if __name__ == "__main__":
    s = AbstractState()
    s[5] = 'hi'
    print('s', s, sep='\t')

    t = AbstractState()
    t[5] = 'lo'
    t[6] = 'new'
    print('t', t, sep='\t')

    print('union', t.union_state(s), sep='\t')
    print('intersect t and union(s,t)', t.intersect_state(t.union_state(s)), sep='\t')