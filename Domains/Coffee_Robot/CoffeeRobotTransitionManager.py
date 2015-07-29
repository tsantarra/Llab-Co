from managers.StateTransitionManager import StateTransitionManager
from graph.State import State
from graph.StateDistribution import StateDistribution

class CoffeeRobotTransitionManager(StateTransitionManager):
    """description of class"""

    def initialState(self, collapse=False):
        return State({'H':False, 'C':True, 'W':False, 'R':True, 'U':False, 'O':True, 'Round':0})

    def transition(self, state, action=None, collapse=False):
        newState = state.copy()
        
        if action is 'Go':
            #Move to other location.
            newState['O'] = not newState['O']

            #Check if the robot gets wet.
            if newState['R'] and not newState['U']:
                newState['W'] = True

        elif action is 'BuyCoffee':
            #If in in coffee shop, buy coffee.
            if newState['O'] == False:
                newState['C'] = True

        elif action is 'DeliverCoffee':
            #If in office and have coffee, deliver coffee.
            if newState['O'] and newState['C']:
                newState['H'] = True
                newState['C'] = False

        elif action is 'GetUmbrella':
            #If in office, pick up umbrella.
            if newState['O']:
                newState['U'] = True

        newState['Round'] += 1

        return StateDistribution(newState)

    def end(self, state):
        """
        Returns True if the scenario has reached an end state.
        """
        return state['H'] or (state['Round'] == 10)
