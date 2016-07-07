"""
Basic interface for an agent holding beliefs about the behavior of its teammate. For now, the base
implementation will use 'Trial-based Heuristic Tree Search for Finite Horizon MDPs' for planning, though
a future version may accept any solver with a custom backup function (required for predicting the actions
of the teammate via the model).
"""
from collections import defaultdict
from functools import partial
import logging

from mdp.distribution import Distribution
from mdp.state import State
from mdp.solvers.thts_dp import graph_search


class ModelingAgent:
    def __init__(self, scenario, identity, models, heuristic=None):
        # Modify scenario with agent-specific adjustments (via function wrappers).
        self.scenario = scenario._replace(transition=modeler_transition(scenario.transition))

        self.identity = identity
        self.models = State(models)
        self.heuristic = heuristic
        self.policy_graph_root = None
        self.policy_backup = partial(policy_backup, agent=self.identity)

    def get_action(self, state):
        local_state = state.update({'Models': self.models})
        (action, node) = graph_search(state=local_state,
                                      scenario=self.scenario,
                                      iterations=1000,
                                      backup_op=self.policy_backup,
                                      heuristic=self.heuristic,
                                      root_node=self.policy_graph_root)
        self.policy_graph_root = node
        return action

    def update(self, agent, state, observation, new_state):
        # Update model
        if agent in self.models:
            new_model = self.models[agent].update(state, observation)
            self.models = self.models.update({agent: new_model})

        new_state = new_state.update({'Models': self.models})
        # Update location in policy graph
        for node in self.policy_graph_root.successors[observation]:
            if node.state == new_state:
                self.policy_graph_root = node
                break
        else:
            raise ValueError(
                "Observation not consistent with predicted transitions." +
                "\nObs: {0}\nNew state: \n{1}\nSuccessors: \n{2}".format(
                    str(observation), str(new_state), str(self.policy_graph_root.successors)))


def policy_backup(node, agent):
    """
    Function given to graph search planner to backup mdp state values based on
        - the agent's expectation maximization process
        - the agent's expectations of teammates' policies
    """
    agent_turn = node.state['Turn']
    #node.value = node.immediate_value
    if node.successors:
        # Calculate expected return for each action at the given node
        action_values = defaultdict(float)
        for action in node.successors:
            action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
            action_values[action] /= sum(node.successors[action].values())

        if agent_turn == agent:  # Agent maximized expectation
            node.value = node.immediate_value + max(action_values.values())
        elif agent_turn in node.state['Models']:  # Agent predicts action distribution and resulting expected value
            action_distribution = node.state['Models'][agent_turn].predict(node.state)
            node.value = node.immediate_value + action_distribution.expectation(action_values, require_exact_keys=False)


"""
Wrappers for scenario from the position of the modeling agent.
"""


def modeler_transition(transition_fn):
    def new_transition_fn(state, action):
        # Get basic information
        agent_turn = state['Turn']

        # Base scenario updates state variables. Models are appended to state but are left unchanged.
        resulting_states = transition_fn(state, action)

        if agent_turn in state['Models']:
            # Update models in resulting states.
            new_resulting_states = Distribution()
            for resulting_state, probability in resulting_states.items():
                new_model = state['Models'][agent_turn].update(state, action)
                new_model_state = resulting_state['Models'].update({agent_turn: new_model})
                resulting_state = resulting_state.update({'Models': new_model_state})
                new_resulting_states[resulting_state] = probability

            return new_resulting_states
        else:
            return resulting_states

    return new_transition_fn
