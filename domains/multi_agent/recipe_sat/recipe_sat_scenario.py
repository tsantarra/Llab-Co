from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace

from random import sample
from itertools import combinations
from math import factorial


class RecipeScenario:

    def __init__(self, num_conditions, num_agents, num_valid_recipes, recipe_size=None):
        self.num_conditions = num_conditions
        self.all_conditions = frozenset(range(num_conditions))
        self.agents = ['Agent' + str(i) for i in range(1, num_agents+1)]
        self.recipes = [set(recipe) for recipe in
                        self.__make_recipes(num_valid_recipes, self.all_conditions, recipe_size)]
        self.success_util = 10
        self.conflict_penalty = 1
        self.extra_cond_penalty = 2

    def initial_state(self):
        """ Gives the initial state of the scenario. """
        return State({'Conditions': frozenset(),
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
        new_state_dict['Conditions'] = new_state_dict['Conditions'] | action_set

        # Return all possible outcomes and their associated probabilities.
        return Distribution({State(new_state_dict): 1.0})

    def utility(self, old_state, action, new_state):
        """ Returns the utility of transitioning from one state to another. """

        # Calculate conflicts
        all_set_conditions = [val for val in action.values() if val != 'End']
        num_overlap = len(all_set_conditions) - len(set(all_set_conditions))
        util = -1 * num_overlap * self.conflict_penalty

        # if not complete, just return conflict util
        if not new_state['Complete']:
            return util

        # check for complete recipe
        set_conditions = new_state['Conditions']
        for recipe in self.recipes:
            if len(recipe - set_conditions) == 0:
                return util + self.success_util - self.extra_cond_penalty * (len(set_conditions) - len(recipe))
        else:
            return util

    def end(self, state):
        """ Returns True if the scenario has reached an end state. """
        return state['Complete']

    def __make_recipes(self, num_recipes, conditions, recipe_length=None):
        """ Generate some subset of combinations that can be used as recipes. """
        if recipe_length:
            return sample(list(combinations(conditions, recipe_length)), num_recipes)

        # otherwise, choose length of recipe that covers num_recipes
        num_combs = lambda n, r: factorial(n)/(factorial(r) * factorial(n-r))
        recipe_length = 1

        # increase recipe length until num recipes is satisfied
        while recipe_length <= len(conditions)/2:
            if num_combs(len(conditions), recipe_length) >= num_recipes:
                return sample(list(combinations(conditions, recipe_length)), num_recipes)

            recipe_length += 1

        raise ValueError('Not enough conditions to generate {num_recipes} recipes.'.format(num_recipes=num_recipes))
