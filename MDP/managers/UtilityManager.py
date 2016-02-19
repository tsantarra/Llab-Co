from abc import ABCMeta, abstractmethod


class UtilityManager:
    """Manages the utility function of a scenario."""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get_utility(self, state):
        """
        Returns the utility associated with being in a state.
        """
        pass



