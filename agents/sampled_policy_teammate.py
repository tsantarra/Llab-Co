from domains.multi_agent.assembly.assembly import get_node_set
from mdp.graph_planner import search
from mdp.distribution import Distribution
from math import exp

global_scenario = None
global_graph = None


def plan(scenario):
    global global_scenario, global_graph

    global_scenario = scenario
    global_graph = search(scenario.initial_state(), scenario, 100000)


def action_probabilities(node, rationality, identity):
    """
    Given an association of values with all joint actions available, return the expectation over each individual
    agent action.
    """
    joint_action_space = node.action_space
    joint_action_values = node.action_values()

    joint_action_probabilities = Distribution({joint_action: exp(rationality * value)
                                               for joint_action, value in joint_action_values.items()})
    joint_action_probabilities.normalize()

    individual_action_probabilities = {action: 0 for action in joint_action_space.individual_actions(identity)}
    for joint_action, prob in joint_action_probabilities.items():
        individual_action_probabilities[joint_action[identity]] += prob

    assert abs(sum(individual_action_probabilities.values()) - 1) <= 10e-5, 'Probabilities do not add up.'

    return Distribution(individual_action_probabilities)


def sample_policy(scenario, rationality, identity):
    global global_graph

    if not global_graph:
        plan(scenario)

    policy = {}
    nodes = get_node_set(global_graph)

    for node in nodes:
        policy[node.state] = action_probabilities(node, rationality, identity).sample()

    return policy


class SampledPolicyTeammate:

    def __init__(self, identity, scenario, rationality=1.0, min_graph_iterations=6):
        """ """
        self.identity = identity
        self.scenario = scenario
        self.rationality = rationality
        self.min_graph_iterations = min_graph_iterations

        # Variables for calculating the policy and storing results
        self.fixed_policy = sample_policy(scenario, rationality, identity)

    def copy(self):
        new_teammate = SampledPolicyTeammate(self.identity, self.scenario, self.rationality, self.min_graph_iterations)
        new_teammate.fixed_policy = self.fixed_policy.copy()
        return new_teammate

    def get_action(self, state):
        return self.predict(state=state, prune=True).sample()

    def predict(self, state, prune=False):
        # Already commited to an action.
        if state not in self.fixed_policy:
            # Perform planning step.
            node = search(state, self.scenario, self.min_graph_iterations, prune=prune)

            # Calculate probabilities for actions.
            action_probs = self.action_probabilities(node)
            self.fixed_policy[state] = action_probs.sample()
            actions = action_probs.keys()
        else:
            actions = self.scenario.actions(state).individual_actions(self.identity)

        return Distribution({action: 1.0 if action == self.fixed_policy[state] else 0
                             for action in actions})

    def action_probabilities(self, node):
        """
        Given an association of values with all joint actions available, return the expectation over each individual
        agent action.
        """
        joint_action_space = node.action_space
        joint_action_values = node.action_values()

        joint_action_probabilities = Distribution({joint_action: exp(self.rationality * value)
                                                   for joint_action, value in joint_action_values.items()})
        joint_action_probabilities.normalize()

        individual_action_probabilities = {action: 0 for action in joint_action_space.individual_actions(self.identity)}
        for joint_action, prob in joint_action_probabilities.items():
            individual_action_probabilities[joint_action[self.identity]] += prob

        assert abs(sum(individual_action_probabilities.values()) - 1) <= 10e-5, 'Probabilities do not add up.'

        return Distribution(individual_action_probabilities)

    def update(self, old_state, observation):
        """ Not recursive. Does not update. May someday wish to update and prune graph. """
        pass

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))