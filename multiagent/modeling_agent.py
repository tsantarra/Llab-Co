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
from mdp.action import Action

from functools import reduce
from operator import mul
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
        # Already updated or no need to update (nothing changed).
        if node.state == new_state:
            return

        # First, update the node's state.
        node.state = new_state

        # End condition. Note: policy and node value won't change for leaf nodes.
        if not node.successors:
            return

        # Update all individual agent models
        individual_agent_actions = node.individual_agent_actions
        model_state = new_state['Models']
        world_state = new_state['World State']
        resulting_models = {agent_name:
                                {action: model_state[agent_name].update(world_state, action) for action in
                                 agent_actions}
                            for agent_name, agent_actions in individual_agent_actions.items()}

        for successor_node, joint_action in [(succ_node, action) for action, succ_dist in node.successors.items()
                                             for succ_node in succ_dist]:
            # Construct new model state from individual agent models
            new_model_state = State({agent_name: resulting_models[agent_name][joint_action[agent_name]]
                               for agent_name in model_state})

            # Calculate new successor state
            old_succ_state = successor_node.state
            new_succ_state = old_succ_state.update({'Models': new_model_state})
            node.successor_transition_values[(new_succ_state, joint_action)] = node.successor_transition_values.pop(
                (old_succ_state, joint_action))

            self.update_policy_graph(successor_node, new_succ_state)

        # After sub-tree/graph is complete, update node values.
        policy_backup(node, self.identity)

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
        joint_action_values = node.calculate_action_values()

        all_joint_actions = list(joint_action_values.keys())

        agent_individual_actions = node.individual_agent_actions

        other_agent_predictions = {other_agent: other_agent_model.predict(node.state['World State'])
                                   for other_agent, other_agent_model in node.state['Models'].items()}

        agent_action_values = {action: 0 for action in agent_individual_actions[agent]}
        for agent_action in agent_individual_actions[agent]:
            all_joint_actions_with_fixed_action = [action for action in all_joint_actions
                                                   if action[agent] == agent_action]

            for joint_action in all_joint_actions_with_fixed_action:
                probability_of_action = reduce(mul, [other_agent_action_dist[joint_action[other_agent]]
                                                     for other_agent, other_agent_action_dist in
                                                     other_agent_predictions.items()])
                agent_action_values[agent_action] += probability_of_action * joint_action_values[joint_action]

        node.future_value = max(agent_action_values.values())


"""
Wrappers for scenario from the position of the modeling agent.
"""


def modeler_transition(transition_fn):
    def new_transition_fn(modeler_state, joint_action):
        # Pull out world state and model state to work with independently
        old_world_state = modeler_state['World State']
        old_model_state = modeler_state['Models']

        # Use scenario's transition function to generate new world states
        resulting_states = transition_fn(old_world_state, joint_action)

        resulting_combined_state_distribution = Distribution()
        for new_world_state, probability in resulting_states.items():
            # Update all models with individual actions from joint_action
            new_model_state = old_model_state.update({agent_name:
                                                          old_model_state[agent_name].update(old_world_state,
                                                                                             joint_action[agent_name])
                                                      for agent_name in old_model_state})

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
