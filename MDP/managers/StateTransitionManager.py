from abc import ABCMeta, abstractmethod

class StateTransitionManager(object):
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def initial_state(self):
        """
        Return the initial state.
        """
        pass

    @abstractmethod
    def transition( self, state, action = None, all_outcomes = False ):
        """
        Given an action in state, return the resulting state or
        a list of all possible outcomes and their associated probabilities
        """
        pass

    @abstractmethod
    def end( self, state):
        """
        Returns whether or not the scenario has ended.
        """
        pass
