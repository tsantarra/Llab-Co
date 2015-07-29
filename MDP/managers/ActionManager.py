from abc import ABCMeta, abstractmethod

class ActionManager(object):
    """Manages actions. Primarily used for returning available actions given a state."""
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def getActions(self, state):
        """
        Returns legal actions given the state.
        """
        pass
