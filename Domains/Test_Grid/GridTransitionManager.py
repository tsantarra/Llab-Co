from managers.StateTransitionManager import StateTransitionManager
from graph.State import State
from graph.StateDistribution import StateDistribution

class GridTransitionManager(StateTransitionManager):
    """description of class"""

    def __init__(self,w,h,x,y):
        self.h = h
        self.w = w
        self.x = x
        self.y = y

    def initialState(self, collapse=False):
        s = State()
        s['x'] = 0
        s['y'] = 0
        return s

    def transition(self, state, action=None, collapse=False):
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
        s = StateDistribution(s) #returning state distributions now
        return s

    def end(self, state):
        return state['Round']==20 or (state['x'] == self.x and state['y'] == self.y)
