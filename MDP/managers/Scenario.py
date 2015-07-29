from MDP.managers.ActionManager import ActionManager
from MDP.managers.StateTransitionManager import StateTransitionManager
from MDP.managers.UtilityManager import UtilityManager

class Scenario(object):
    """Organizes the operations of a scenario through manager."""

    def __init__(self, stateTransMngr, actionMngr, utilityMngr):
        """
        Assigns the managers to class variables.
        """
        self.STM = stateTransMngr
        self.AM = actionMngr
        self.UM = utilityMngr

    def transitionState(self, state, action=None):
        """
        Progresses the state given an action. Returns resulting state.
        """
        return self.STM.transition(state, action)

    def getActions(self, state):
        """
        Returns the actions available in the given state.
        """
        return self.AM.getActions(state)

    def getUtility(self, state):
        """
        Returns the utility associated with the given state.
        """
        return self.UM.getUtility(state)

    def end(self, state):
        """
        Returns whether or not the scenario has ended.
        """
        return self.STM.end(state)