from managers.UtilityManager import UtilityManager

class GridUtilityManager(object):
    """Manages utility for GridTest"""

    def __init__(self,x,y):
        """
        Sets goal state.
        """
        self.x = x
        self.y = y

    def getUtility(self, state):
        """
        Returns utility associated with given state.
        """
        utility = int(state['x'] == self.x and state['y'] == self.y)
        return utility


