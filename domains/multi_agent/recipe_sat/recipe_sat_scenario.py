from collections import Counter, namedtuple
from collections import defaultdict

from mdp.scenario import Scenario
from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace

RecipeAction = namedtuple('RecipeAction', ['act', 'object'])
RecipeAction.__new__.__defaults__ = (None,) * len(RecipeAction._fields)  # Sets default values to None


class RecipeScenario:

    def __init__(self, num_components=3, num_recipes=5, rounds=10, ingredients_per_recipe=5,
                 ingredients='ABCDEFGHIGJKLMNOPQRSTUVWXYZ'):
        self.possible_components = ingredients[0:num_components]
        self.recipes = set()  # TODO
        self.util_per_recipe = ingredients_per_recipe
        self.rounds = rounds

        self.agents = ['Agent1', 'Agent2']

    def initial_state(self):
        """
        Gives the initial state of the scenario.
            - Empty inventory for each agent
            - Round
        """
        return State({'Inventory 1': ItemCounts({comp: 0 for comp in self.possible_components}),
                      'Inventory 2': ItemCounts({comp: 0 for comp in self.possible_components}),
                      'Round': 0,
                      'Complete': False})

    def actions(self, state):
        """ Returns legal actions in the state. """
        agent_actions = {agent: [RecipeAction('Wait', None)] for agent in self.agents}
        if state['Complete'] or state['Round'] >= self.rounds:
            return JointActionSpace(agent_actions)

        for agent in self.agents:
            action_list = [RecipeAction('Prepare', ingredient) for ingredient in self.possible_components]

            if agent == 'Agent2':
                teammate_inventory = state['Inventory 2']
                construct_actions = [RecipeAction('Construct', recipe) for recipe in self.recipes
                                if all(teammate_inventory[ingredient] >= count for ingredient, count in recipe.items())]
                action_list += construct_actions

                action_list += [RecipeAction('Accept')]

            else:
                action_list += [RecipeAction('Pass', ingredient) for ingredient in self.possible_components
                                if state['Inventory 1'][ingredient] > 0]

            agent_actions[agent] += action_list

        return JointActionSpace(agent_actions)

    def transition(self, state, action):
        """
        Cases to handle:
            - preparing materials and updating inventories
            - passing items
            - constructing an object from a recipe
            - the round
        """
        new_state_dict = dict(state.copy())

        agent1_action = action['Agent1']
        agent2_action = action['Agent2']

        inventory1 = dict(new_state_dict['Inventory 1'])
        inventory2 = dict(new_state_dict['Inventory 2'])

        # Agent1 actions
        if 'Prepare' == agent1_action.act:
            item = agent1_action.object
            inventory1[item] += 2
        elif 'Pass' == agent1_action.act and 'Accept' == agent2_action.act:
            item = agent1_action.object
            inventory2[item] += inventory1[item]
            inventory1[item] = 0

        # Agent2 actions
        if 'Prepare' == agent2_action.act:
            item = agent2_action.object
            inventory2[item] += 1
        elif 'Construct' == agent2_action.act:
            recipe = agent2_action.object
            for item, count in recipe.items():
                inventory2[item] -= count
            new_state_dict['Complete'] = True

        new_state_dict['Inventory 1'] = ItemCounts(inventory1)
        new_state_dict['Inventory 2'] = ItemCounts(inventory2)
        new_state_dict['Round'] += 1

        return Distribution({State(new_state_dict): 1.0})  # all possible outcomes and their associated probabilities

    def utility(self, old_state, action, new_state):
        """
        Returns the utility of transitioning from one state to another.
        """
        util = 0
        if 'Construct' == action['Agent2'].act:
            util += self.util_per_recipe

        agent_actions = action.values()
        any_pass = any('Pass' == action.act for action in agent_actions)
        any_accept = any('Accept' == action.act for action in agent_actions)

        # penalize for having any pass or accept without having the other
        if any_pass != any_accept:
            util -= 1

        if new_state['Complete'] or new_state['Round'] >= self.rounds:
            # end penalty for extra materials
            util -= sum(new_state['Inventory 1'].values()) + sum(new_state['Inventory 2'].values())

        return util

    def end(self, state):
        """
        Returns True if the scenario has reached an end state.
        """
        return state['Round'] >= self.rounds or state['Complete']
