from mdp.graph_planner import search, map_graph
from mdp.distribution import Distribution

from math import exp, inf
from collections import defaultdict


class UniformPolicyTeammate:

    def __init__(self, identity, scenario):
        """ """
        self.identity = identity
        self.scenario = scenario

    def copy(self):
        new_teammate = UniformPolicyTeammate(self.identity, self.scenario)
        new_teammate.state_actions = self.state_actions.copy()
        return new_teammate

    def get_action(self, state):
        raise Exception('Residual policy teammate should never be asked for a next action (get_action).')

    def predict(self, state):
        actions = self.scenario.actions(state).individual_actions(self.identity)
        return Distribution({action: 1.0/len(actions) for action in actions})

    def update(self, old_state, observation):
        """ Not recursive. Does not update. May someday wish to update and prune graph. """
        pass

    def __eq__(self, other):
        return isinstance(other, UniformPolicyTeammate)

    def __hash__(self):
        return hash(self.identity)