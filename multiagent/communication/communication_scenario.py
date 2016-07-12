from mdp.distribution import Distribution
from mdp.scenario import Scenario
from mdp.state import State
from mdp.graph_planner import *


class CommScenario:

    def __init__(self, policy_graph):
        # Save references to module variables
        self._policy_root = policy_graph

        self._policy_ev_cache = {}
        self._base_policy = {}
        self._modeler_states = set()
        self._teammate_states = set()
        self._model = None

    def _calculate_ev(self, policy_state):
        pass

    def initial_state(self):

        # Collect policy info
        policy_states = map_tree(self._policy_root)

        # Construct policy state
        state = State({})

        return None

    def actions(self, policy_state):
        pass

    def transition(self, policy_state, query):
        pass

    def end(self, policy_state):
        pass

    def utility(self, old_policy_state, query, new_policy_state):
        if old_policy_state not in self._policy_ev_cache:
            pass  # TODO

        if new_policy_state not in self._policy_ev_cache:
            pass  # TODO

        return self._policy_ev_cache[new_policy_state] - self._policy_ev_cache[old_policy_state]


