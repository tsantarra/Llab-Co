from abc import ABCMeta, abstractmethod

class StateTransitionManager(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def initialState( self, collapse = False ):
        """
        Return the initial state.
        """
        pass

    @abstractmethod
    def transition( self, state, action = None, collapse = False ):
        """
        Given an action in state, return the resulting state or distribution
        of states. 
        """
        pass

    @abstractmethod
    def end( self, state):
        """
        Returns whether or not the scenario has ended.
        """
        pass