from MDP.managers.ActionManager import ActionManager

class GridActionManager(ActionManager):
    """Manages actions for grid test."""

    def __init__(self, w, h):
        """Initialize grid test constraints."""
        self.h = h
        self.w = w

    def get_actions(self, state):
        """Returns legal actions in the state."""
        actions = []

        if state['x'] > 0:
            actions += ['left']
        if state['x'] < self.w:
            actions += ['right']
        if state['y'] > 0:
            actions += ['down']
        if state['y'] < self.h:
            actions += ['up']

        return actions

