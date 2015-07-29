from abc import ABCMeta, abstractmethod


class UtilityManager(object):
    """Manages the utility function of a scenario."""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def getUtility(self, state):
        """
        Returns the utility associated with being in a state.
        """
        pass



