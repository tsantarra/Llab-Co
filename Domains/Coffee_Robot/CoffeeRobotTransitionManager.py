from MDP.managers.StateTransitionManager import StateTransitionManager
from MDP.graph.State import State

class CoffeeRobotTransitionManager(StateTransitionManager):
    """description of class"""

    def initial_state(self):
        state = State({'H':False, 'C':True, 'W':False, 'R':True, 'U':False, 'O':True})
        state['Round'] = 0
        return state

    def transition(self, state, action=None, all_outcomes=False):
        new_state = state.copy()
        
        if action is 'Go':
            # Move to other location.
            new_state['O'] = not new_state['O']

            # Check if the robot gets wet.
            if new_state['R'] and not new_state['U']:
                new_state['W'] = True

        elif action is 'BuyCoffee':
            # If in in coffee shop, buy coffee.
            if not new_state['O']:
                new_state['C'] = True

        elif action is 'DeliverCoffee':
            # If in office and have coffee, deliver coffee.
            if new_state['O'] and new_state['C']:
                new_state['H'] = True
                new_state['C'] = False

        elif action is 'GetUmbrella':
            # If in office, pick up umbrella.
            if new_state['O']:
                new_state['U'] = True

        new_state['Round'] += 1

        if not all_outcomes:
            return new_state
        else:
            return [(new_state, 1.)] # a list of all possible outcomes and their associated probabilities

    def end(self, state):
        """
        Returns True if the scenario has reached an end state.
        """
        return state['H'] or (state['Round'] == 10)

