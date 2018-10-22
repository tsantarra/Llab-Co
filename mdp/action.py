"""
What are the requirements of a joint action?
    - should refer to agents by name; Map?
    - create a helper static method to generate all possible joint actions from individual agent actions

"""

from collections import Mapping
from itertools import product


class Action(Mapping):
    def __init__(self, agent_action_dict):
        self._dict = agent_action_dict.copy()
        self._hash = None

    def update(self, agent_actions):
        new_dict = self._dict.copy()
        new_dict.update(agent_actions)
        return Action(new_dict)

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return 'Action(' + str(self._dict) + ')'

    def __eq__(self, other):
        return self._dict == other._dict

    def __lt__(self, other):
        return tuple(sorted(self._dict.items())) < tuple(sorted(other._dict.items()))

    def __hash__(self):
        if not self._hash:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash


class JointActionSpace:
    def __init__(self, individual_agent_actions):
        self.agent_actions = {name: actions if type(actions) is set else set(actions)
                              for name, actions in individual_agent_actions.items()}

    def individual_actions(self, agent_name=None):
        return self.agent_actions[agent_name] if agent_name else self.agent_actions

    def fix_actions(self, fixed_actions):
        assert all(key in self.agent_actions for key in fixed_actions), 'Missing agent while fixing actions:' + \
                str(fixed_actions) + ' ' + str(self.agent_actions)

        return JointActionSpace({**self.agent_actions, **fixed_actions})

    def constrain(self, fixed_agent_actions):
        new_individual_actions = self.agent_actions.copy()
        new_individual_actions.update(fixed_agent_actions)

        return JointActionSpace(new_individual_actions)

    def __iter__(self):
        action_lists = [[(name, action) for action in action_list] for name, action_list in self.agent_actions.items()]
        return (Action(dict(combination)) for combination in product(*action_lists))

    def __str__(self):
        return str(self.agent_actions)

