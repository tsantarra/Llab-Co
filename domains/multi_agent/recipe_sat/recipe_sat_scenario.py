from collections import namedtuple

from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace


class Conditions(State):
    def __str__(self): return ' '.join(str(item) for item in self.items())

    def __repr__(self): return str(self)


class RecipeScenario:

    def __init__(self, num_conditions, num_agents, num_valid_recipes, recipe_size):
        self.num_conditions = num_conditions
        self.all_conditions = set(range(num_conditions))
        self.agents = ['Agent' + str(i) for i in range(1, num_agents+1)]
        self.recipes = self.__make_recipes(num_valid_recipes, recipe_size)
        self.success_util = 10
        self.conflict_penalty = 1

    def initial_state(self):
        """ Gives the initial state of the scenario. """
        return State({'Conditions': set(),
                      'Complete': False,
                      })

    def actions(self, state):
        """ Returns legal actions in the state. """
        if state['Complete']:
            agent_actions = {agent: [] for agent in self.agents}
            return JointActionSpace(agent_actions)

        # Otherwise, set actions for each remaining condition.
        unset_conditions = self.all_conditions - state['Conditions']
        agent_actions = {agent: list(unset_conditions) + ['End'] for agent in self.agents}

        return JointActionSpace(agent_actions)

    def transition(self, state, action):
        """  """
        new_state_dict = dict(state.copy())
        action_set = set(action.values())  # all unique actions

        # Mark completion
        if 'End' in action_set:
            new_state_dict['Complete'] = True
            action_set -= {'End'}

        # Update conditions
        new_state_dict['Conditions'] |= action_set

        # Return all possible outcomes and their associated probabilities.
        return Distribution({State(new_state_dict): 1.0})

    def utility(self, old_state, action, new_state):
        """ Returns the utility of transitioning from one state to another. """
        util = 0

        # Calculate conflicts
        all_set_conditions = [val for val in action.values() if val != 'End']
        num_overlap = len(all_set_conditions) - len(set(all_set_conditions))
        util -= num_overlap * self.conflict_penalty

        # if not complete, just return conflict util
        if not new_state['Complete']:
            return util
        else:
            # check for complete recipe
            set_conditions = new_state['Conditions']
            if any(len(recipe - set_conditions) == 0 for recipe in self.recipes):
                return util + self.success_util
            else:
                return util

    def end(self, state):
        """ Returns True if the scenario has reached an end state. """
        return state['Complete']

    def __make_recipes(self, num_recipes, recipe_size):
        return []
