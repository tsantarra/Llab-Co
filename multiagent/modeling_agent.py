"""
Basic interface for an agent holding beliefs about the behavior of its teammate. For now, the base
implementation will use 'Trial-based Heuristic Tree Search for Finite Horizon MDPs' for planning, though
a future version may accept any solver with a custom backup function (required for predicting the actions
of the teammate via the model).
"""
from functools import partial

from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import search
from copy import copy


class ModelingAgent:
    def __init__(self, scenario, identity, models, heuristic=None):
        # Modify scenario with agent-specific adjustments (via function wrappers).
        if isinstance(scenario, tuple):  # is a namedtuple
            self.scenario = scenario._replace(transition=modeler_transition(scenario.transition),
                                              actions=modeler_actions(scenario.actions),
                                              end=modeler_end(scenario.end),
                                              utility=modeler_utility(scenario.utility)
                                              )
        else:  # is a class
            self.scenario = copy(scenario)
            self.scenario.transition = modeler_transition(self.scenario.transition)
            self.scenario.actions = modeler_actions(self.scenario.actions)
            self.scenario.end = modeler_end(self.scenario.end)
            self.scenario.utility = modeler_utility(self.scenario.utility)

        self.identity = identity
        self.model_state = State(models)
        self.heuristic = heuristic
        self.policy_graph_root = None
        self.policy_backup = partial(policy_backup, agent=self.identity)

    def get_action(self, state):
        local_state = State({'World State': state, 'Models': self.model_state})

        (action, node) = search(state=local_state,
                                scenario=self.scenario,
                                iterations=1000,
                                backup_op=self.policy_backup,
                                heuristic=self.heuristic,
                                root_node=self.policy_graph_root)
        self.policy_graph_root = node
        return action

    def update_policy_graph(self, node, new_state):
        """
        Items to update:
            - node states
            - transition utils
            - node values
        """
        # First, update the node's state.
        node.state = new_state

        # End condition. Note: policy and node value won't change for leaf nodes.
        if not node.successors:
            return

        # Update successors
        agent_turn = node.state['World State']['Turn']
        model_state = new_state['Models']
        for successor_node, action in [(succ_node, action) for action, succ_dist in node.successors.items()
                                       for succ_node in succ_dist]:
            # update models
            if agent_turn in model_state:
                new_model = model_state[agent_turn].update(node.state['World State'], action)
                new_model_state = model_state.update({agent_turn: new_model})
            else:
                new_model_state = new_state['Models']

            # Calculate new successor state
            old_succ_state = successor_node.state
            new_succ_state = old_succ_state.update({'Models': new_model_state})
            node.successor_transition_values[(new_succ_state, action)] = node.successor_transition_values.pop(
                (old_succ_state, action))

            self.update_policy_graph(successor_node, new_succ_state)

        # Update node value
        action_values = node.calculate_action_values()
        agent_turn = node.state['World State']['Turn']
        if agent_turn not in node.state['Models']:
            action, value = max(action_values.items(), key=lambda pair: pair[1])
            node.future_value = value
        else:  # Agent predicts action distribution and resulting expected value
            action_distribution = node.state['Models'][agent_turn].predict(node.state['World State'])
            node.future_value = action_distribution.expectation(action_values)

    def update(self, agent_name, old_state, observation, new_state):
        # Update model
        if agent_name in self.model_state:
            new_model = self.model_state[agent_name].update(old_state, observation)
            self.model_state = self.model_state.update({agent_name: new_model})

        if self.policy_graph_root:  # Can't update if the agent has not planned yet
            new_modeler_state = State(
                {'World State': new_state, 'Models': self.model_state})  # new_state.update({'Models': self.models})

            # Update location in policy graph
            for node in self.policy_graph_root.successors[observation]:
                if node.state == new_modeler_state:
                    self.policy_graph_root = node
                    break
            else:
                # Every successor should be included in the graph.
                raise ValueError('Successor state not found in modeling agent update of policy graph.')


"""
Helper functions for planning.
"""


def policy_backup(node, agent):
    """
    Function given to graph search planner to backup mdp state values based on
        - the agent's expectation maximization process
        - the agent's expectations of teammates' policies
    """
    if node.successors:
        # Calculate expected return for each action at the given node
        action_values = node.calculate_action_values()

        agent_turn = node.state['World State']['Turn']
        if agent_turn == agent:  # Agent maximized expectation
            node.future_value = max(action_values.values())
        elif agent_turn in node.state['Models']:  # Agent predicts action distribution and resulting expected value
            action_distribution = node.state['Models'][agent_turn].predict(node.state['World State'])
            node.future_value = action_distribution.expectation(action_values)


"""
Wrappers for scenario from the position of the modeling agent.
"""


def modeler_transition(transition_fn):
    def new_transition_fn(modeler_state, action):
        # Pull out world state and model state to work with independently
        old_world_state = modeler_state['World State']
        old_model_state = modeler_state['Models']

        # Get basic information
        agent_turn = old_world_state['Turn']

        # Use scenario's transition function to generate new world states
        resulting_states = transition_fn(old_world_state, action)

        resulting_combined_state_distribution = Distribution()
        for new_world_state, probability in resulting_states.items():
            if agent_turn in old_model_state:
                # Update models in resulting states.
                old_teammate_model = old_model_state[agent_turn]
                new_teammate_model = old_teammate_model.update(old_world_state, action)
                new_model_state = old_model_state.update({agent_turn: new_teammate_model})
            else:
                new_model_state = old_model_state.copy()

            resulting_combined_state = State({'World State': new_world_state, 'Models': new_model_state})
            resulting_combined_state_distribution[resulting_combined_state] = probability

        return resulting_combined_state_distribution

    return new_transition_fn


def modeler_end(end_fn):
    def new_end(modeler_state):
        return end_fn(modeler_state['World State'])

    return new_end


def modeler_utility(utility_fn):
    def new_utility(old_modeler_state, action, new_modeler_state):
        return utility_fn(old_modeler_state['World State'] if old_modeler_state else None,
                          action,
                          new_modeler_state['World State'])

    return new_utility


def modeler_actions(actions_fn):
    def new_actions(modeler_state):
        return actions_fn(modeler_state['World State'])

    return new_actions
