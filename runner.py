__author__ = 'Trevor'

from MDP.graph.State import State

if __name__ == "__main__":
    a = State(dict(zip([1,2,3,4], 'abcd')))
    b = State(dict(zip([1,2,3,4], 'abcz')))
    c = a.feature_intersection(b)


    d = State(dict(zip([1,2], 'ab')))

    print(d in c)
