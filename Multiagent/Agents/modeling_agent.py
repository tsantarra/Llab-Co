"""
Basic interface for an agent holding beliefs about the behavior of its teammate. For now, the base
implementation will use 'Trial-based Heuristic Tree Search for Finite Horizon MDPs' for planning, though
a future version may accept any solver with a custom backup function (required for predicting the actions
of the teammate via the model).

TODO: generalize solvers, low priority
TODO: wrap planner in such a way to take into account future observations of teammate (POMDP)
"""
from collections import defaultdict
from functools import partial

from MDP.Distribution import Distribution
from MDP.solvers.thts_dp import graph_search
from Multiagent.Agent import Agent


class ModelingAgent(Agent):

    def __init__(self, scenario, identity, models, heuristic=None):
        # Modify scenario with agent-specific adjustments (via function wrappers).
        self.scenario = scenario._replace(transition=modeler_transition(scenario.transition))

        self.identity = identity
        self.models = models
        self.heuristic = heuristic
        self.policy_graph = None
        self.policy_backup = partial(policy_backup, agent=self.identity)

    def get_action(self, state):
        state['Models'] = self.models
        (action, node) = graph_search(state=state,
                                      scenario=self.scenario,
                                      iterations=1000,
                                      backup_op=self.policy_backup,
                                      heuristic=self.heuristic,
                                      root_node=self.policy_graph)
        self.policy_graph = node
        return action

    def update(self, agent, state, observation, new_state):
        # Update model
        if agent in self.models:
            self.models[agent] = self.models[agent].update(state, observation)

        # Update location in policy graph
        for node in self.policy_graph.successors[observation]:
            if node.state == new_state:
                self.policy_graph = node
                break


def policy_backup(node, agent):
    """
    Function given to graph search planner to backup MDP state values based on
        - the agent's expectation maximization process
        - the agent's expectations of teammates' policies
    """
    agent_turn = node.state['Turn']
    node.value = node.immediate_value
    if node.successors:
        # Calculate expected return for each action at the given node
        action_values = defaultdict(float)
        for action in node.successors:
            action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
            action_values[action] /= sum(node.successors[action].values())

        if agent_turn == agent:  # Agent maximized expectation
            node.value += max(action_values.values())
        elif agent_turn in node.state['Models']:  # Agent predicts action distribution and resulting expected value
            action_distribution = node.state['Models'][agent_turn].predict(node.state)
            node.value += action_distribution.expectation(action_values)


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
                resulting_state['Models'][agent_turn] = resulting_state['Models'][agent_turn].update(state, action)
                new_resulting_states[resulting_state] = probability

            return new_resulting_states
        else:
            return resulting_states

    return new_transition_fn


