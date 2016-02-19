from MDP.managers.ActionManager import ActionManager
from MDP.managers.StateTransitionManager import StateTransitionManager
from MDP.managers.UtilityManager import UtilityManager

class Scenario:
    """Organizes the operations of a scenario through manager."""

    def __init__(self, stateTransMngr, actionMngr, utilityMngr):
        """
        Assigns the managers to class variables.
        """
        self.STM = stateTransMngr
        self.AM = actionMngr
        self.UM = utilityMngr

    def initial_state(self):
        """
        Returns the initial state of the scenario.
        """
        return self.STM.initial_state()

    def transition_state(self, state, action=None, all_outcomes=False):
        """
        Progresses the state given an action. Returns resulting state.
        """
        return self.STM.transition(state, action, all_outcomes)

    def get_actions(self, state):
        """
        Returns the actions available in the given state.
        """
        return self.AM.get_actions(state)

    def get_utility(self, state):
        """
        Returns the utility associated with the given state.
        """
        return self.UM.get_utility(state)

    def end(self, state):
        """
        Returns whether or not the scenario has ended.
        """
        return self.STM.end(state)