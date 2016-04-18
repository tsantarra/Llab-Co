"""
Displays a directed, acyclic graph using the networkx library (for graph data types), matplotlib
(rendering), and the graphviz algorithm (for minimizing edge crosses).

Algorithm paper: http://www.graphviz.org/Documentation/TSE93.pdf
Paper title/authors: A Technique for Drawing Directed Graphs - Gansner, Koutsofios, North, and Vo

Note: If pygraphviz updates to Python 3.5, networkx supplies a wrapper call for layout, making this
unnecessary.
"""


import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, deque


def _add_edges(node, graph, graph_map, node_set, horizon=0):
    """
    Traverses the graph, storing nodes in the graph_map.
    """
    node_set.add(node)
    graph_map[horizon].append(node)
    for successor in [successor_dist for dist in node.successors.values() for successor_dist in dist]:
        graph.add_edge(node, successor)
        if successor not in node_set:
            _add_edges(successor, graph, graph_map, node_set, horizon+1)


def _median_order(graph, go_up):
    order = sorted(graph, reverse=go_up)
    for horizon in order:
        median_values = {}
        for node in graph[horizon]:
            median_values[node] = _median_value(node, graph, horizon, go_up)

        node_order = sorted(median_values, key=lambda n: median_values[n])
        graph[horizon] = {node: index for index, node in enumerate(node_order)}


def _median_value(v, graph, horizon, go_up):
    if go_up:
        # going up the graph/tree; want to calculate based on reordered children
        if (horizon+1) not in graph:
            return graph[horizon][v]
        adj_node_ranks = sorted(graph[horizon+1][succ] for succ_dist in v.successors.values() for succ in succ_dist)
    else:
        # going down the graph/tree; want to calculate based on reordered parents
        if (horizon-1) not in graph:
            return graph[horizon][v]
        adj_node_ranks = sorted(graph[horizon-1][pred] for pred in v.predecessors)

    num_adj = len(adj_node_ranks)
    median_index = int(num_adj/2)
    if num_adj == 0:
        return graph[horizon][v]
    elif num_adj % 2 == 1:
        # use median
        return adj_node_ranks[median_index]
    elif num_adj == 2:
        # average the two
        return (adj_node_ranks[0] + adj_node_ranks[1])/2
    else:
        # do something smart based on how things are skewed?
        left = adj_node_ranks[median_index-1] - adj_node_ranks[0]
        right = adj_node_ranks[-1] - adj_node_ranks[median_index]
        return (adj_node_ranks[median_index-1] * right + adj_node_ranks[median_index] * left)/(left + right)


def _transpose(graph):
    """
    Iterates over graph ordering greedily swapping successive pairs if it reduces crosses.
    """
    improved = True
    while improved:
        improved = False
        for horizon, level in graph.items():
            current_order = sorted(level, key=lambda x: level[x])

            for i in range(0, len(current_order)-1):
                v = current_order[i]
                w = current_order[i+1]
                if _should_swap(v, w, graph, horizon):
                    graph[horizon][v], graph[horizon][w] = graph[horizon][w], graph[horizon][v]
                    current_order[i], current_order[i+1] = current_order[i+1], current_order[i]
                    improved = True


def _crosses(graph_map):
    """
    Counts the total number of edge crosses in the current graph.
    """
    total_crosses = 0
    for horizon in graph_map:
        ordered_nodes = sorted(graph_map[horizon], key=lambda n: graph_map[horizon][n])
        for i in range(len(ordered_nodes) - 1):
            v = ordered_nodes[i]
            for j in range(i+1, len(ordered_nodes)):
                w = ordered_nodes[j]
                if (horizon + 1) in graph_map:
                    for v_succ in [succ for succ_dist in v.successors.values() for succ in succ_dist]:
                        for w_succ in [succ for succ_dist in w.successors.values() for succ in succ_dist]:
                            total_crosses += int(graph_map[horizon + 1][w_succ] < graph_map[horizon + 1][v_succ])
    return total_crosses


def _should_swap(v, w, graph_map, horizon):
    """
    Counts the number of edge crosses if v appears left of w and vice versa. Determines if less crosses
    would occur if v and w were switched.
    """
    left_crosses = 0
    right_crosses = 0
    if (horizon-1) in graph_map:
        for v_pred in v.predecessors:
            for w_pred in w.predecessors:
                left_crosses += int(graph_map[horizon-1][w_pred] < graph_map[horizon-1][v_pred])
                right_crosses += int(graph_map[horizon-1][v_pred] < graph_map[horizon-1][w_pred])

    if (horizon+1) in graph_map:
        for v_succ in [succ for succ_dist in v.successors.values() for succ in succ_dist]:
            for w_succ in [succ for succ_dist in w.successors.values() for succ in succ_dist]:
                left_crosses += int(graph_map[horizon+1][w_succ] < graph_map[horizon+1][v_succ])
                right_crosses += int(graph_map[horizon+1][v_succ] < graph_map[horizon+1][w_succ])

    # return True if more crosses in current order than in swapped order.
    return left_crosses > right_crosses


def _median(values):
    """
    Finds the median of a set of values.
    """
    values = sorted(values)
    num = len(values)
    median = int(num/2)
    return values[median] if (num % 2 == 1) else (values[median-1] + values[median])/2


def _median_pos(xcoords, up):
    """
    Adjusts the x coordinates of nodes to the medians of their
        - downward neighbors, if going down
        - upward neighbors, if going up

    Note: unlike the ordering process, this adjusts based on the next level, not the previous (and already
    adjusted) level.
    """
    for horizon in sorted(xcoords, reverse=up):
        for node in xcoords[horizon]:
            if up:
                targets = node.predecessors
                target_horizon = horizon-1
            else:
                targets = [succ for succ_dist in node.successors.values() for succ in succ_dist]
                target_horizon = horizon+1

            if targets:
                xcoords[horizon][node] = _median(xcoords[target_horizon][target] for target in targets)


def _min_node(xcoords, up):
    """
    Performs local optimization one node at a time, using a queue. Initially all nodes are queued. When
    a node is removed from the queue, it is placed as close as possible to the median of all its neighbors
    (both up and down). If the node’s placement is changed, its neighbors are re-queued if not already in
    the queue. _min_node terminates when it achieves a local minimum
    """
    nodes = [(node, horizon) for horizon, node_level in xcoords.items() for node in node_level]
    queue = deque(nodes)  # appendleft, pop
    in_queue = set(nodes)

    while queue:
        node, horizon = queue.pop()

        top_neighbors = [(neighbor, horizon-1) for neighbor in node.predecessors]
        bottom_neighbors = [(neighbor, horizon+1) for succ_dist in node.successors.values() for neighbor in succ_dist]
        all_neighbors = top_neighbors + bottom_neighbors
        neighbor_coords = [xcoords[h][neighbor] for neighbor, h in all_neighbors]

        new_median = _median(neighbor_coords)
        if xcoords[horizon][node] != new_median:
            xcoords[horizon][node] = new_median

            to_queue = [neighbor_tuple for neighbor_tuple in all_neighbors if neighbor_tuple not in in_queue]
            for neighbor_tuple in to_queue:
                queue.appendleft(neighbor_tuple)
                in_queue.add(neighbor_tuple)


def _normalize(xcoords):
    """
    In place of packcut. To combat drift, shift all node coordinates back to the left by an offset of min node coord.

    Note: Initial version won't keep a minimum distance between nodes in a level. Hopefully we won't need
    to consider that.
    """
    pass


def _xlength(xcoords):
    """
    Returns the sum of edge widths (not lengths, just x component distances).  This is the function the
    process should be minimizing, forming more vertical edges.
    """
    total = 0
    for horizon in xcoords:
        for node, x in xcoords[horizon].items():
            total += sum(abs(x - xcoords[horizon+1][succ]) for succ_dist in node.successors.values() for succ in succ_dist)

    return total


def show_graph(root):
    width = 7
    height = 7
    graph = nx.MultiDiGraph()  # unlike digraph, supports multiple parallel edges between nodes
    graph_map = defaultdict(list)

    # Steps to position DAG
    # Traverse graph add nodes by horizon level (rank)
    _add_edges(root, graph, graph_map, set())

    graph_map = {rank: {node: index for index, node in enumerate(graph_map[rank])} for rank in graph_map}

    ###########################################################
    best_map = graph_map.copy()
    # Iterate n times (default 24)
    for i in range(24):
        # Reorder by median heuristic
        _median_order(graph_map, i % 2)

        # Greedy transpose
        _transpose(graph_map)

        # Save best
        if _crosses(graph_map) < _crosses(best_map):
            best_map = graph_map.copy()

    graph_map = best_map

    ############################################################
    # X coordinate positioning
    x_coords = graph_map.copy()
    positions = {}

    max_horizon = max(x_coords.keys())
    for horizon in range(max_horizon+1):
        nodes = x_coords[horizon]
        num_nodes = len(nodes)
        for node, i in nodes.items():
            x_coords[horizon][node] = (i/num_nodes) * width

    best = x_coords.copy()
    for i in range(8):  # 8 iterations suggested
        up = (i % 2 == 0)
        _median_pos(x_coords, up)
        _min_node(x_coords, up)
        #_normalize(x_coords)
        if _xlength(x_coords) < _xlength(best):
            best = x_coords.copy()
    x_coords = best

    #############################################
    for horizon in x_coords:
        for node, x in x_coords[horizon].items():
            positions[node] = (x, height - (horizon/max_horizon) * height)

    node_values = [node.value for node in graph.nodes()]
    max_value = max(node_values)
    node_values = [val/max_value for val in node_values]

    plt.figure(figsize=(8, 8))
    nx.draw(graph, cmap=plt.get_cmap('plasma'), pos=positions, node_color=node_values)#, with_labels=True)
    plt.show()

