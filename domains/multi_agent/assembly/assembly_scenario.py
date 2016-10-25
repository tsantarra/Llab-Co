from collections import Counter, namedtuple
from mdp.scenario import Scenario
from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace

from random import choice

AssemblyAction = namedtuple('AssemblyAction', ['act', 'object'])
AssemblyAction.__new__.__defaults__ = (None,) * len(AssemblyAction._fields)  # Sets default values to None


class ItemCounts(State):
    def __str__(self): return ' '.join(str(item) for item in self.items())

    def __repr__(self): return str(self)


def make_recipe(components, total):
    """
    Create a recipe made of 'components' such that it requires 'total' individual components.
    """
    recipe = Counter()
    for _ in range(total):
        recipe[choice(components)] += 1

    return ItemCounts(recipe.items())


ingredients = 'ABCDEFGHIGJKLMNOPQRSTUVWXYZ'
possible_components = ingredients[0:3]
rounds = 10
num_recipes = 3
ingredients_per_recipe = 3
recipes = [make_recipe(possible_components, ingredients_per_recipe) for _ in range(num_recipes)]
print(recipes)
agents = ['Agent1', 'Agent2']


def initial_state():
    """
    Gives the initial state of the scenario.
        - Empty inventory for each agent
        - Round
    """
    return State({'Inventory 1': ItemCounts({comp: 0 for comp in possible_components}),
                  'Inventory 2': ItemCounts({comp: 0 for comp in possible_components}),
                  'Round': 0})


def actions(state):
    """ Returns legal actions in the state. """
    agent_actions = {}

    for agent in agents:
        action_list = [AssemblyAction('Prepare', ingredient) for ingredient in possible_components]

        if agent == 'Agent2':
            teammate_inventory = state['Inventory 2']
            action_list += [AssemblyAction('Construct', recipe) for recipe in recipes
                            if all(teammate_inventory[ingredient] >= count for ingredient, count in recipe.items())]
            action_list += [AssemblyAction('Accept')]

        else:
            action_list += [AssemblyAction('Pass', ingredient) for ingredient in possible_components
                            if state['Inventory 1'][ingredient] > 0]

        agent_actions[agent] = action_list

    return JointActionSpace(agent_actions)


def transition(state, action):
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
        inventory1[item] += 1
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

    new_state_dict['Inventory 1'] = ItemCounts(inventory1)
    new_state_dict['Inventory 2'] = ItemCounts(inventory2)
    new_state_dict['Round'] += 1

    return Distribution({State(new_state_dict): 1.0})  # all possible outcomes and their associated probabilities


def utility(old_state, action, new_state):
    """
    Returns the utility of transitioning from one state to another.
    """
    util = 0
    if 'Construct' == action['Agent2'].act:
        util += ingredients_per_recipe

    agent_actions = action.values()
    any_pass = any('Pass' == action.act for action in agent_actions)
    any_accept = any('Accept' == action.act for action in agent_actions)

    # penalize for having any pass or accept without having the other
    if any_pass != any_accept:
        util -= 1

    return util


def end(state):
    """
    Returns True if the scenario has reached an end state.
    """
    return state['Round'] >= rounds


assembly_scenario = Scenario(initial_state=initial_state, actions=actions, transition=transition, utility=utility,
                             end=end)
