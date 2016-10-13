
from random import randint, choice, shuffle
from collections import Counter
from mdp.scenario import Scenario
from mdp.distribution import Distribution
from mdp.state import State


def make_recipe(components, total):
    recipe = Counter()
    for _ in range(total):
        recipe[choice(components)] += 1

    return recipe

ingredients = 'ABCDEFGHIGJKLMNOPQRSTUVWXYZ'
possible_components = ingredients[0:3]
rounds = 20
num_recipes = 3
ingredients_per_recipe = 3
recipes = [make_recipe(possible_components, ingredients_per_recipe) for _ in range(num_recipes)]


def initial_state():
    """
    """
    return State({'Inventory 1': {comp: 0 for comp in possible_components},
                  'Inventory 2': {comp: 0 for comp in possible_components},
                  'Recipes Made': {recipe: 0 for recipe in recipes},
                  'Round': 0,
                  'Turn': 'Agent',
                  'Passed?': None})


def transition(state, action):
    new_state_dict = dict(state.copy())

    if 'Prepare' in action:
        [item] = [ing for ing in possible_components if ing in action]
        if new_state_dict['Turn'] == 'Agent':
            new_state_dict['Inventory 1'][item] += 1
        else:
            new_state_dict['Inventory 2'][item] += 1

    elif 'Construct' in action:
        [recipe] = [recipe for recipe in recipes if recipe in action]
        new_state_dict['Recipes Made'][recipe] += 1

    elif 'Accept' in action:
        if new_state_dict['Passed'] is not None:
            item = new_state_dict['Passed?']
            new_state_dict['Inventory 1'][item] -= 1
            new_state_dict['Inventory 2'][item] += 1
            new_state_dict['Passed'] = None

    elif 'Pass' in action:
        [item] = [ing for ing in ingredients if ing in action]
        new_state_dict['Passed'] = item

    return Distribution({State(new_state_dict): 1.0})  # all possible outcomes and their associated probabilities


def actions(state):
    """ Returns legal actions in the state. """
    action_list = ['Prepare ' + ingredient for ingredient in possible_components]
    # Add construction of recipes
    if state['Turn'] == 'Teammate':
        teammate_inventory = state['Inventory 2']
        action_list += ['Construct ' + str(recipe) for recipe in recipes
                    if all(teammate_inventory[ingredient] >= recipe[ingredient] for ingredient in ingredients)]
        action_list += ['Accept ' + ingredient for ingredient in possible_components]
    else:
        action_list += ['Pass ' + ingredient for ingredient in possible_components
                        if state['Inventory 1'][ingredient] > 0]

    return action_list


def utility(old_state, action, new_state):
    """
    Returns the utility of transitioning from one state to another.
    """
    util = 0
    if 'Construct' in action:
        util += ingredients_per_recipe

    return util


def end(state):
    """
    Returns True if the scenario has reached an end state.
    """
    return state['Round'] >= rounds


assembly_scenario = Scenario(initial_state=initial_state, actions=actions, transition=transition, utility=utility, end=end)