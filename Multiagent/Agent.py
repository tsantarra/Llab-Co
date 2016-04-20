"""
Basic interface for an agent holding beliefs about the behavior of its teammate. For now, the base
implementation will use 'Trial-based Heuristic Tree Search for Finite Horizon MDPs' for planning, though
a future version may accept any solver with a custom backup function (required for predicting the actions
of the teammate via the model).

TODO: generalize solvers, low priority
TODO: wrap planner in such a way to take into account future observations of teammate (POMDP)
"""


class Agent:

    def get_action(self, state):
        raise NotImplementedError

    def update(self, state, observation, agent, new_state):
        raise NotImplementedError