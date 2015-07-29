from MDP.managers.ActionManager import ActionManager

class CoffeeRobotActionManager(ActionManager):
    """Manages actions for grid test."""

    def __init__(self):
        pass

    def getActions(self, state):
        """Returns legal actions in the state."""
        actions = ['Go','BuyCoffee','DeliverCoffee','GetUmbrella']

        return actions

