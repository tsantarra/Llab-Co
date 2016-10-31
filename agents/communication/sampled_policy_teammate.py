from mdp.graph_planner import search
from mdp.distribution import Distribution
from math import exp


class SampledPolicyTeammate:

    def __init__(self, identity, scenario, rationality=1.0, min_graph_size=100):
        """ """
        self.identity = identity
        self.scenario = scenario
        self.rationality = rationality
        self.min_graph_size = min_graph_size

        # Variables for calculating the policy and storing results
        self.fixed_policy = {}
        self.state_node_map = {}
        self.roots = set()

    def get_action(self, state):
        return self.predict(state=state, prune=True).sample()

    def predict(self, state, prune=False):
        # Already commited to an action.
        if state in self.fixed_policy:
            return Distribution({self.fixed_policy[state]: 1.0})

        # Need to find the correspond graph node or create a new one.
        if state in self.state_node_map:
            node = self.state_node_map[state]
            self.roots -= node.predecessors
            self.roots.add(node)
            iterations = max(self.min_graph_size - node.visits, 0)  # in the event node.visits > min_graph_size
        else:
            node = None
            iterations = self.min_graph_size

        # Perform planning step.
        node = search(state, self.scenario, iterations, root_node=node, prune=prune)

        # Calculate probabilities for actions.
        action_probs = self.action_probabilities(node)
        self.fixed_policy[state] = action_probs.sample()

        # Update class vars for new nodes.
        if state not in self.state_node_map:
            self.state_node_map[state] = node
            self.roots.add(node)

        # Map nodes for lookup later.
        self.map_graphs()

        return Distribution({self.fixed_policy[state]: 1.0})

    def action_probabilities(self, node):
        """
        Given an association of values with all joint actions available, return the expectation over each individual
        agent action. We're just going to assume
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

    def check_nodes(self):
        """
        Sanity check: make sure pruning and mapping is accurate.
        """
        process_list = list(self.roots)
        processed_set = set()

        while process_list:
            # Grab a node to process.
            node = process_list.pop()

            # Don't repeat nodes.
            if node in processed_set:
                continue

            # Add child nodes and continue.
            if node.successors:
                process_list += list(node.successor_set())

            # Add to processed set
            processed_set.add(node)

        print('Total nodes:', len(processed_set))
        print('Mapped nodes:', len(self.state_node_map))

    def map_graphs(self):
        process_list = list(self.roots)

        self.state_node_map = {}
        processed_set = set()

        while process_list:
            # Grab a node to process.
            node = process_list.pop()

            # Don't repeat nodes.
            if node in processed_set:
                continue

            # If there are two nodes that correspond to the same state, keep the one with more visits.
            state = node.state
            if state in self.state_node_map:
                other_node = self.state_node_map[state]
                self.state_node_map[state] = max([node, other_node], key=lambda n: n.visits)
            else:
                self.state_node_map[state] = node

            # Add child nodes and continue.
            if node.successors:
                process_list += list(node.successor_set())

            # The node is now processed
            processed_set.add(node)

    def update(self, old_state, observation):
        """ Not recursive. Does not update. May someday wish to update and prune graph. """
        pass

    def clear_graphs(self):
        self.state_node_map = {}
        self.roots = {}

    def __eq__(self, other):
        return self.fixed_policy == other.fixed_policy
