from collections import defaultdict, deque


def map_graph_by_depth(root):
    """ Traverse the graph. Return a mapping of node to horizon."""
    process_list = deque([root])
    depth_map = defaultdict(lambda: 0)
    depth_map[root] = 0

    # Queue all nodes in tree according to depth
    while process_list:
        node = process_list.pop()
        horizon = depth_map[node]

        for joint_action, successor_distribution in node.successors.items():
            for successor in successor_distribution:
                if horizon + 1 > depth_map[successor]:
                    # The successor can be deeper in certain branches.
                    depth_map[successor] = horizon + 1
                    process_list.append(successor)

    return depth_map


def create_node_set(node, node_set=None):
    if node_set is None:
        node_set = set()

    open_set = {node}
    while open_set:
        node = open_set.pop()
        node_set.add(node)
        open_set |= node.successor_set() - node_set

    return node_set


def detect_cycle(node):
    """
    Uses DFS to detect if there exist any cycles in the directed graph.
    """

    # Define recursive depth-first search function
    def dfs(current, unvisited, incomplete, complete):
        unvisited.remove(current)
        incomplete.add(current)

        for successor in current.successor_set():
            if successor in complete:
                continue
            elif successor in incomplete:
                return True
            elif dfs(successor, unvisited, incomplete, complete):
                return True

        incomplete.remove(current)
        complete.add(current)
        return False

    # Return result (all nodes are reachable from the node, so only one call is necessary)
    return dfs(node, create_node_set(node), set(), set())


def map_graph(node):
    """
    Builds a dict mapping of states to corresponding nodes in the graph.
    """
    return {n.state: n for n in create_node_set(node)}


def prune_unreachable(node, node_map, checked):
    """
    Prunes currently unreachable nodes from the graph, which cuts down on policy computation time for
    irrelevant areas of the state space.
    """
    checked.add(node)
    prune_set = set(pred for pred in node.predecessors if pred.state not in node_map)
    node.predecessors -= prune_set

    for pruned_node in prune_set:
        pruned_node.successors = {}

    for successor in [succ for succ in node.successor_set() if succ not in checked]:
        prune_unreachable(successor, node_map, checked)


def traverse_graph_topologically(depth_map, node_fn, top_down=True):
    """ Traverses the graph either top-down or bottom-up. """
    factor = 1 if top_down else -1
    node_list = list((factor * horizon, node) for node, horizon in depth_map.items())
    node_list.sort()

    for priority, node in node_list:
        node_fn(node, priority * factor)


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    import sys
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

