from MDP.managers.StateTransitionManager import StateTransitionManager
from MDP.graph.State import State

class GridTransitionManager(StateTransitionManager):
    """description of class"""

    def __init__(self,w,h,x,y):
        self.h = h
        self.w = w
        self.x = x
        self.y = y

    def initial_state(self):
        s = State()
        s['x'] = 0
        s['y'] = 0
        s['Round'] = 0
        return s

    def transition(self, state, action=None, all_outcomes=False):
        s = state.copy()
        
        if action is 'right':
            s['x'] = s['x']+1
        elif action is 'left':
            s['x'] = s['x']-1
        elif action is 'up':
            s['y'] = s['y']+1
        elif action is 'down':
            s['y'] = s['y']-1

        s['Round'] += 1

        if not all_outcomes:
            return s
        else:
            return [(s, 1.)] # a list of all possible outcomes and their associated probabilities

    def end(self, state):
        return state['Round'] == 20 or (state['x'] == self.x and state['y'] == self.y)
