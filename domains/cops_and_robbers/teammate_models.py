from collections import defaultdict
from functools import partial
from heapq import heappop, heappush

from mdp.distribution import Distribution
from multiagent.experts_model import ExpertsModel

(WALL, OPEN, AGENT, PARTNER, ROBBER, GATE_UP, GATE_DOWN, GATE_RIGHT, GATE_LEFT) = \
    ('*', ' ', 'A', 'S', 'R', '^', 'v', '>', '<')


def a_star(start, end, maze):
    """ Performs A* from start to end in the graph. Returns a list of tuples corresponding
        to the path found.
    """
    heuristic_estimates = defaultdict(float)
    path_cost = defaultdict(float)
    parents = {}

    closed_set = set()
    open_set = {start}
    open_heap = [(0, start)]

    current = None
    while open_set:
        cost, current = heappop(open_heap)
        if current == end:
            break

        open_set.remove(current)
        closed_set.add(current)

        row, col = current
        neighbors = []
        if maze[(row + 1, col)] == OPEN or maze[(row + 1, col)] == GATE_DOWN:
            neighbors.append((row + 1, col))
        if maze[(row - 1, col)] == OPEN or maze[(row - 1, col)] == GATE_UP:
            neighbors.append((row - 1, col))
        if maze[(row, col + 1)] == OPEN or maze[(row, col + 1)] == GATE_RIGHT:
            neighbors.append((row, col + 1))
        if maze[(row, col - 1)] == OPEN or maze[(row, col - 1)] == GATE_LEFT:
            neighbors.append((row, col - 1))

        for neighbor in neighbors:
            if neighbor in closed_set:
                continue
            elif neighbor in open_set:
                new_path_cost = path_cost[current] + 1
                if new_path_cost < path_cost[neighbor]:
                    path_cost[neighbor] = new_path_cost
                    parents[neighbor] = current
            else:
                path_cost[neighbor] = path_cost[current] + 1
                heuristic_estimates[neighbor] = abs(neighbor[0]-end[0]) + abs(neighbor[1]-end[1])
                parents[neighbor] = current

                open_set.add(neighbor)
                heappush(open_heap, (path_cost[current] + heuristic_estimates[current], neighbor))

    path = []
    while current in parents:
        path.append(current)
        current = parents[current]
    path.append(current)
    return path[::-1]


def get_move(start, end):
    row_diff = end[0] - start[0]
    col_diff = end[1] - start[1]
    if row_diff == 1:
        return 'D'
    elif row_diff == -1:
        return 'U'
    elif col_diff == 1:
        return 'R'
    elif col_diff == -1:
        return 'L'
    else:
        return 'W'


def init_map_cache(maze):
    width = max(key[0] for key in maze)
    height = max(key[1] for key in maze)

    path_cache = defaultdict(lambda x=None: ('W', None, width * height))  # Move, the next cell, distance
    pool = set((start, end) for start in maze for end in maze if (maze[start] != WALL and maze[end] != WALL))

    while pool:
        start, end = pool.pop()
        path = a_star(start, end, maze)

        for i, begin in enumerate(path):

            if i != len(path)-1:
                next_cell = path[i+1]
                move = get_move(begin, next_cell)

                for j, loc in enumerate(path[i+1:]):
                    path_cache[(begin, loc)] = (move, next_cell, j-i)
                    if (begin, loc) in pool:
                        pool.remove((begin, loc))

            else:
                path_cache[(begin, begin)] = ('W', begin, 0)

        if not path:
            path_cache[(start, end)] = ('W', start, width * height)

    return path_cache


def a_star_predict(state, target, actions, map_cache):
    partner_loc, target_loc = state['P'], state[target]
    action_list = actions(state)

    action_probs = Distribution({action: 0.01 for action in action_list})
    best_action = 'P-' + map_cache[(partner_loc, target_loc)][0]
    assert best_action in action_probs
    action_probs[best_action] = 0.99
    action_probs.normalize()

    return action_probs


# Used for constructing the models of the teammate from the ad hoc coordinator's perspective.
def build_experts_model(scenario, maze, initial_state):
    map_cache = init_map_cache(maze)
    predictors = Distribution({partial(a_star_predict,
                                       target=key,
                                       actions=scenario.actions,
                                       map_cache=map_cache): 1
                               for key in initial_state if 'Robber' in key})
    predictors.normalize()

    return ExpertsModel(scenario, predictors)


# The actual teammate
class AstarTeammate:

    def __init__(self, scenario, target, maze):
        self.scenario = scenario
        self.target = target
        self.cached_paths = init_map_cache(maze)

    def get_action(self, state):
        return a_star_predict(state, self.target, self.scenario.actions, self.cached_paths).sample()

    def update(self, agent_name, old_state, observation, new_state):
        pass
