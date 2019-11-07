from collections import defaultdict, namedtuple, deque
from dataclasses import dataclass
from numbers import Number


@dataclass(frozen=True)
class ValueEstimate:
    lower_bound: float = 0
    estimate: float = 0
    upper_bound: float = 0

    def __init__(self, lb, est, ub):
        object.__setattr__(self, 'lower_bound', lb)
        object.__setattr__(self, 'estimate', est)
        object.__setattr__(self, 'upper_bound', ub)

    def __add__(self, other):
        if isinstance(other, ValueEstimate):
            return ValueEstimate(self.lower_bound + other.lower_bound,
                                 self.estimate + other.estimate,
                                 self.upper_bound + other.upper_bound)
        elif isinstance(other, Number):
            return ValueEstimate(self.lower_bound + float(other),
                                 self.estimate + float(other),
                                 self.upper_bound + float(other))
        else:
            raise NotImplementedError(f'__add__ operator not implemented for type {type(other)}.')

    def __mul__(self, other):
        if isinstance(other, Number):
            return ValueEstimate(self.lower_bound * float(other),
                                 self.estimate * float(other),
                                 self.upper_bound * float(other))
        else:
            raise NotImplementedError(f'__mul__ operator not implemented for type {type(other)}.')

    def __rmul__(self, other):
        return self * other


class StateNode:
    def __init__(self, state, terminal, predecessor=None, value=ValueEstimate(0, 0, 0)):
        """
        Initializes tree node with relevant information.
        """
        self.state = state
        self.value = value
        self.terminal = terminal
        self.complete = terminal
        self.action_nodes = None
        self.predecessors = {predecessor} if predecessor else set()

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return f'<Val: {"%.2f" % self.value} {self.complete}>\n{self.state}'

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self is other

    def __del__(self):
        for attr in vars(self).values():
            del attr


ActionResult = namedtuple('ActionResult', ['state_node', 'probability', 'utility'])


class ActionNode:
    def __init__(self, action):
        self.action = action
        self.value = ValueEstimate(0, 0, 0)
        self.complete = False
        self.successors = []

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return f'<Q-Val: {"%.2f" % self.value} {self.complete}>\n{self.action}'

    def __hash__(self):
        return hash(self.action)

    def __eq__(self, other):
        return self is other

    def __del__(self):
        for attr in vars(self).values():
            del attr


class SearchContext:
    def __init__(self, scenario):
        self.scenario = scenario
        self.state_node_lookup = dict()
        self.root = None
        self.depth_map = defaultdict(int)

    def add_state(self, state, predecessor=None):
        node = StateNode(state,
                         terminal=self.scenario.end(state),
                         predecessor=predecessor,
                         value=self.scenario.heuristic(state))
        self.state_node_lookup[state] = node
        return node

    def get_node(self, state):
        return self.state_node_lookup[state]

    def update_root(self, state):
        self.root = self.state_node_lookup[state] if state in self.state_node_lookup else self.add_state(state)
        self.depth_map = defaultdict(int, {state: horizon - 1 for state, horizon in self.depth_map.items() if horizon > 0})


def search(search_context, iterations, budget_per_iteration):
    for iteration in range(iterations):
        selected_nodes = select(search_context, budget_per_iteration)
        expand(search_context, selected_nodes)
        backpropagate(search_context, selected_nodes)


def select(search_context, budget):
    queue = deque([(search_context.root, 0)])
    state_budgets = defaultdict(int, {(search_context.root, 0): budget})
    action_budgets = defaultdict(int)
    selected_nodes = {}

    while queue:
        state_node, horizon = queue.popleft()

        # Check for traversal termination
        if horizon > search_context.depth_map[state_node.state] or not state_node.action_nodes:
            selected_nodes[state_node] = horizon
            continue

        budget = state_budgets[(state_node, horizon)]
        max_lower_bound = max(action_node.value.lower_bound for action_node in state_node.action_nodes)
        action_budgets.clear()

        # Assign 1 to every non-dominated arm (sorted by upper bound)
        non_dominated = sorted((action_node for action_node in state_node.action_nodes
                                if not action_node.complete and action_node.value.upper_bound >= max_lower_bound),
                               key=lambda n: n.value.upper_bound,
                               reverse=True)
        to_assign_1 = min(budget, len(non_dominated))
        action_budgets.update({action_node: 1 for action_node in non_dominated[:to_assign_1]})

        # Assign remaining to max expected action
        max_action_node = max((action_node for action_node in non_dominated), key=lambda node: node.value.estimate)
        action_budgets[max_action_node] += budget - to_assign_1

        next_horizon = horizon + 1
        for action_node, budget in action_budgets.items():
            # Assign 1 to every non-complete state and the remaining to the max of Pr * (ub - lb)
            incomplete_successors = sorted((action_result for action_result in action_node.successors if not action_result.state_node.complete),
                                           key=lambda ar: ar.probability * (ar.state_node.value.upper_bound - ar.state_node.value.lower_bound),
                                           reverse=True)
            to_assign_1 = min(budget, len(incomplete_successors))
            state_budgets[(incomplete_successors[0].state_node, next_horizon)] += budget - to_assign_1 + 1
            for action_result in incomplete_successors[1:to_assign_1]:
                state_budgets[(action_result.state_node, next_horizon)] += 1

    return selected_nodes


def expand(search_context, selected_nodes):
    pass


def backpropagate(search_context, selected_nodes):
    pass


if __name__ == '__main__':
    # test values
    a = ValueEstimate(-1, 0, 4)
    print(3 * a)
    print(a * 3)
    print(a + 3)
    print(a + a)
