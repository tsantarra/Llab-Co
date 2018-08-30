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
        self.__hash = None

    def update(self, agent_actions):
        new_dict = self.__dict.copy()
        new_dict.update(agent_actions)
        return Action(new_dict)

    def __getitem__(self, key):
        return self.__dict[key]

    def __iter__(self):
        return iter(self.__dict)

    def __len__(self):
        return len(self.__dict)

    def __repr__(self):
        return 'Action(' + str(self.__dict) + ')'

    def __eq__(self, other):
        return self.__dict == other.__dict

    def __lt__(self, other):
        return tuple(sorted(self.__dict.items())) < tuple(sorted(other.__dict.items()))

    def __hash__(self):
        if not self.__hash:
            self.__hash = hash(tuple(sorted(self.__dict.items())))
        return self.__hash


class JointActionSpace:
    def __init__(self, individual_agent_actions):
        self.agent_actions = {name: actions if type(actions) is set else set(actions)
                              for name, actions in individual_agent_actions.items()}
        self.joint_actions = self.all_joint_actions()

    def all_joint_actions(self):
        action_lists = [[(name, action) for action in action_list] for name, action_list in self.agent_actions.items()]

        # potentially make this a generator so we do not need to enumerate all joint actions first
        return [Action(dict(combination)) for combination in product(*action_lists)]

    def individual_actions(self, agent_name=None):
        if agent_name:
            return self.agent_actions[agent_name]
        else:
            return self.agent_actions

    def fix_actions(self, fixed_actions):
        assert all(key in self.agent_actions for key in fixed_actions), 'Missing agent while fixing actions:' + \
                str(fixed_actions) + ' ' + str(self.agent_actions)
        return JointActionSpace({**self.agent_actions, **fixed_actions})

    def constrain(self, fixed_agent_actions):
        new_individual_actions = self.agent_actions.copy()
        new_individual_actions.update(fixed_agent_actions)
        return JointActionSpace(new_individual_actions)

    def __iter__(self):
        return iter(self.joint_actions)

    def __getitem__(self, item):
        """ Used for indexing (e.g. actions[2])."""
        return self.joint_actions[item]

    def __len__(self):
        return len(self.joint_actions)

    def __str__(self):
        return str(self.agent_actions)

    def __bool__(self):
        return len(self.joint_actions) != 0
