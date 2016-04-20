"""
Basic interface for an agent holding beliefs about the behavior of its teammate. For now, the base
implementation will use 'Trial-based Heuristic Tree Search for Finite Horizon MDPs' for planning, though
a future version may accept any solver with a custom backup function (required for predicting the actions
of the teammate via the model).

TODO: generalize solvers, low priority
TODO: wrap planner in such a way to take into account future observations of teammate (POMDP)
"""
from MDP.solvers.thts_dp import graph_search
from Multiagent.Agent import Agent

from collections import defaultdict
from functools import partial


def policy_backup(node, agent, models):
    """
    Function given to graph search planner to backup MDP state values based on
        - the agent's expectation maximization process
        - the agent's expectations of teammates' policies
    """
    turn = node.state['Turn']
    node.value = node.immediate_value
    if node.successors:
        # Calculate expected return for each action at the given node
        action_values = defaultdict(float)
        for action in node.successors:
            action_values[action] = sum(child.value * prob for child, prob in node.successors[action].items())
            action_values[action] /= sum(node.successors[action].values())

        if turn == agent:  # Agent maximized expectation
            node.value += max(action_values.values())
        elif turn in models:  # Agent predicts action distribution and resulting expected value
            action_distribution = models[turn].predict(node.state)
            node.value += action_distribution.expectation(action_values)


class ModelingAgent(Agent):

    def __init__(self, scenario, identity, models, heuristic=None):
        self.identity = identity
        self.scenario = scenario
        self.models = models
        self.heuristic = heuristic
        self.policy_graph = None
        self.policy_backup = partial(policy_backup, agent=self.identity, models=self.models)

    def get_action(self, state):
        (action, node) = graph_search(state=state,
                                      scenario=self.scenario,
                                      iterations=1000,
                                      backup_op=self.policy_backup,
                                      heuristic=self.heuristic,
                                      root_node=self.policy_graph)
        self.policy_graph = node
        return action

    def update(self, state, observation, agent, new_state):
        # Update model
        if agent in self.models:
            self.models[agent].update(state, observation)

        # Update location in policy graph
        for node in self.policy_graph.successors[observation]:
            if node.state == new_state:
                self.policy_graph = node
                break



"""
Wrappers for scenario from the position of the modeling agent.
"""

def modeler_transition(transition_fn, agent):

    def new_transition_fn(state, action):
        resulting_states = transition_fn(state, action)
        agent.update(state, action, state['Turn'], ) ############ RESULTING STATE

        #what if we just wrapped all parts of the planner?????????????????????

        return resulting_states

    return new_transition_fn
