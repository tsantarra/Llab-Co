from MDP.managers.UtilityManager import UtilityManager

class CoffeeRobotUtilityManager(UtilityManager):
    """Reports the utility of a state in the Coffee Robot Scenario."""

    def get_utility(self, state):
        """
        Returns the utility of the state.
        """
        utility = 0
        if state['W']:
            utility += -0.5
        if state['H']:
            utility += 1*(0.9)**(state['Round'])

        return utility


