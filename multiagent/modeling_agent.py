"""
Basic interface for an agent holding beliefs about the behavior of its teammate. For now, the base
implementation will use 'Trial-based Heuristic Tree Search for Finite Horizon MDPs' for planning, though
a future version may accept any solver with a custom backup function (required for predicting the actions
of the teammate via the model).
"""
from collections import defaultdict
from functools import partial

from mdp.distribution import Distribution
from mdp.state import State
from mdp.graph_planner import search
from copy import copy


class ModelingAgent:
    def __init__(self, scenario, identity, models, heuristic=None):
        # Modify scenario with agent-specific adjustments (via function wrappers).
        if isinstance(scenario, tuple): # is a namedtuple
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
            #types.MethodType(modeler_transition(self.scenario.transition), self.scenario)

        self.identity = identity
        self.model_state = State(models)
        self.heuristic = heuristic
        self.policy_graph_root = None
        self.policy_backup = partial(policy_backup, agent=self.identity)

    def get_action(self, state):
        local_state = State({'World State': state, 'Models': self.model_state})

        self.policy_graph_root = None
        (action, node) = search(state=local_state,
                                scenario=self.scenario,
                                iterations=1000,
                                backup_op=self.policy_backup,
                                heuristic=self.heuristic,
                                root_node=self.policy_graph_root)
        self.policy_graph_root = node
        return action

    def recalculate_policy(self):
        def compute_policy(node, action_values, new_policy):
            action, action_value = max(action_values.items(), key=lambda pair: pair[1])
            new_policy[node.state] = action
            return action_value

        def traverse_policy_graph(node, node_values, model_state, policy, policy_fn):
            """ Computes a policy maximizing expected utility given a probabilistic agent model for other agents. """
            # Already visited this node. Return its computed value.
            if node in node_values:
                return node_values[node]

            # Leaf node. Value already calculated as immediate + heuristic.
            if not node.successors:
                node_values[node] = node.value
                return node.value

            # Calculate expected return for each action at the given node
            active_agent = node.state['World State']['Turn']
            action_values = defaultdict(float)
            for action, result_distribution in node.successors.items():
                assert abs(sum(result_distribution.values()) - 1.0) < 10e-5, 'Action probabilities do not sum to 1.'
                for resulting_state_node, result_probability in result_distribution.items():
                    if active_agent in model_state:
                        new_model = model_state[active_agent].update(node.state, action)
                        new_model_state = model_state.update({active_agent: new_model})
                    else:
                        new_model_state = model_state

                    action_values[action] += result_probability * traverse_policy_graph(resulting_state_node,
                                                                                        node_values,
                                                                                        new_model_state, policy,
                                                                                        policy_fn)

            # Compute the node value
            if active_agent not in model_state:
                # Given a pre-computed policy, use the associated action
                node.value = node.immediate_value + policy_fn(node, action_values, policy)
                node_values[node] = node.value
            else:
                # Agent predicts action distribution and resulting expected value
                action_distribution = model_state[active_agent].predict(node.state['World State'])

                node.value = node.immediate_value + action_distribution.expectation(action_values,
                                                                                           require_exact_keys=False)
                node_values[node] = node.value

            return node_values[node]

        policy = {}
        traverse_policy_graph(self.policy_graph_root, {}, self.model_state, policy, compute_policy)

    def update(self, agent_name, old_state, observation, new_state):
        # Update model
        if agent_name in self.model_state:
            new_model = self.model_state[agent_name].update(old_state, observation)
            self.model_state = self.model_state.update({agent_name: new_model})

        if __name__ == '__main__':
            if self.policy_graph_root: # Can't update if the agent has not planned yet
                new_modeler_state = State(
                    {'World State': new_state, 'Models': self.model_state})  # new_state.update({'Models': self.models})

                # Update location in policy graph
                for node in self.policy_graph_root.successors[observation]:
                    if node.state == new_modeler_state:
                        self.policy_graph_root = node
                        break
                else:
                    # Could have new model after communicating. Therefore, start planning anew.
                    self.policy_graph_root = None


def policy_backup(node, agent):
    """
    Function given to graph search planner to backup mdp state values based on
        - the agent's expectation maximization process
        - the agent's expectations of teammates' policies
    """
    agent_turn = node.state['World State']['Turn']
    if node.successors:
        # Calculate expected return for each action at the given node
        action_values = defaultdict(float)
        for action in node.successors:
            action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
            action_values[action] /= sum(node.successors[action].values())

        if agent_turn == agent:  # Agent maximized expectation
            node.value = node.immediate_value + max(action_values.values())
        elif agent_turn in node.state['Models']:  # Agent predicts action distribution and resulting expected value
            action_distribution = node.state['Models'][agent_turn].predict(node.state['World State'])
            node.value = node.immediate_value + action_distribution.expectation(action_values, require_exact_keys=False)


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