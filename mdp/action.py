
"""
What are the requirements of a joint action?
    - should refer to agents by name; Map?
    - create a helper static method to generate all possible joint actions from individual agent actions

"""

from collections import Mapping
from itertools import product


class Action(Mapping):

    def __init__(self, agent_action_dict):
        self.__dict = agent_action_dict.copy()

    @staticmethod
    def all_joint_actions(individual_action_lists):
        agent_names = individual_action_lists.keys()
        ordered_action_lists = [individual_action_lists[name] for name in agent_names]

        return [Action(dict(zip(agent_names, combination)))
                for combination in product(*ordered_action_lists)]

    @staticmethod
    def all_individual_actions(list_of_joint_actions):
        agent_names = list(list_of_joint_actions[0].keys())

        return {name: set(action[name] for action in list_of_joint_actions) for name in agent_names}

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __repr__(self):
        return 'Action: ' + str(self.__dict)




