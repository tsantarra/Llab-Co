"""
This scenario is taken from an AIIDE paper by Owen Macindoe et al. It is a variant of the traditional
multiagent pursuit problem, where a team of agents pursues a fleeing prey on a toroidal grid. Here, two
agents coordinate to trap one (of several) prey in a maze.

State variables:
    Agent position
    Partner position
    Prey positions
    Round

The state can also be extended with a partner model, taking the role of the 'belief state' in POMDPs. 
"""

from random import shuffle

from mdp.distribution import Distribution
from mdp.scenario import Scenario
from mdp.state import State

(WALL, OPEN, AGENT, PARTNER, ROBBER, GATE_UP, GATE_DOWN, GATE_RIGHT, GATE_LEFT) = \
    ('*', ' ', 'A', 'S', 'R', '^', 'v', '>', '<')

maze = {}


def initialize_maze(maze_file):
    with open(maze_file, 'r') as mazeFile:
        maze_text = mazeFile.read()

    maze_lines = maze_text.split('\n')

    global maze
    maze = {(row, col): char for row, line in enumerate(maze_lines) for col, char in enumerate(line)}
    return maze


def initial_state():
    global maze
    state_dict = dict()

    robber_count = 0
    for loc, char in maze.items():
        if char == AGENT:
            state_dict['A'] = loc
            maze[loc] = OPEN
        elif char == PARTNER:
            state_dict['P'] = loc
            maze[loc] = OPEN
        elif char == ROBBER:
            robber_count += 1
            state_dict['Robber' + str(robber_count)] = loc
            maze[loc] = OPEN

    state_dict['Round'] = 1
    state_dict['Turn'] = 'A'

    return State(state_dict)


def show_state(state):
    robbers = [rob for rob in state if 'Robber' in rob]
    players = [('A', 'A'), ('P', 'P')] + [(rob, 'R') for rob in robbers]

    maze_copy = maze.copy()
    for key, char in players:
        maze_copy[state[key]] = char

    string = ''
    for row in range(1 + max(k[0] for k in maze_copy.keys())):
        for col in range(1 + max(k[1] for k in maze_copy.keys())):
            string += maze_copy[(row, col)]
        string += '\n'

    string += str([state['A'], state['P'], state['Round']]) + '\n'
    return string


def actions(state):
    global maze
    prefix = state['Turn']
    row, col = state[prefix]

    legal_actions = [prefix + '-W']
    if maze[(row + 1, col)] == OPEN or maze[(row + 1, col)] == GATE_DOWN:
        legal_actions.append(prefix + '-D')
    if maze[(row - 1, col)] == OPEN or maze[(row - 1, col)] == GATE_UP:
        legal_actions.append(prefix + '-U')
    if maze[(row, col + 1)] == OPEN or maze[(row, col + 1)] == GATE_RIGHT:
        legal_actions.append(prefix + '-R')
    if maze[(row, col - 1)] == OPEN or maze[(row, col - 1)] == GATE_LEFT:
        legal_actions.append(prefix + '-L')

    return legal_actions


def move_robbers(state):
    global maze
    result = Distribution({state: 1.0})

    agent = state['A']
    partner = state['P']
    for robber, loc in [(ID, loc) for ID, loc in state.items() if ID.startswith('Robber')]:
        row, col = loc
        targets = [loc]
        if maze[(row + 1, col)] == OPEN or maze[(row + 1, col)] == GATE_DOWN:
            targets.append((row + 1, col))
        if maze[(row - 1, col)] == OPEN or maze[(row - 1, col)] == GATE_UP:
            targets.append((row - 1, col))
        if maze[(row, col + 1)] == OPEN or maze[(row, col + 1)] == GATE_RIGHT:
            targets.append((row, col + 1))
        if maze[(row, col - 1)] == OPEN or maze[(row, col - 1)] == GATE_LEFT:
            targets.append((row, col - 1))
        shuffle(targets)

        # using Manhattan distance for convenience
        def distance(p1, p2): return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        distances = [(t, distance(t, agent), distance(t, partner)) for t in targets]

        # find the best target maximizing the minimum distance to the players
        best_target = max(distances, key=lambda x: min(x[1], x[2]))
        ties = [d for d in distances if min(d[1], d[2]) == min(best_target[1], best_target[2])]

        # break ties on distance to just agent
        best_target = max(ties, key=lambda x: x[1])
        ties = [d for d in ties if d[1] == best_target[1]]

        new_state_dist = Distribution()
        for result_state, state_prob in result.items():
            tie_prob = 1/len(ties)
            for target, d1, d2 in ties:
                new_state = result_state.update({robber: target})  # state.update() returns a modified copy
                new_state_dist[new_state] = tie_prob * state_prob
        result = new_state_dist

    assert abs(sum(result.values()) - 1.0) < 10e-5, 'Resulting transition too large. ' + str(sum(result.values()))

    return result


def transition(state, action):
    new_state_diff = {}
    actor, direction = action.split('-')

    row, col = state[actor]
    if direction == 'D':
        row += 1
    elif direction == 'U':
        row -= 1
    elif direction == 'R':
        col += 1
    elif direction == 'L':
        col -= 1

    new_state_diff[actor] = (row, col)
    new_state_diff['Round'] = state['Round'] + 1

    if actor == 'P':
        new_state_diff['Turn'] = 'A'
        new_state = state.update(new_state_diff)
        if not end(new_state):
            # Check if end first, otherwise impossible (robbers always escape).
            return move_robbers(new_state)
        else:
            return Distribution({new_state: 1.0})
    else:
        new_state_diff['Turn'] = 'P'
        new_state = state.update(new_state_diff)
        return Distribution({new_state: 1.0})


def end(state):
    if state['Round'] >= 100:
        return True
    a_loc, p_loc = state['A'], state['P']
    for loc in [loc for ID, loc in state.items() if 'Robber' in ID]:
        if a_loc == p_loc == loc:
            return True
    return False


def utility(old_state, action, new_state):
    return (100 - new_state['Round']) if end(new_state) else 0


def heuristic(modeler_state):
    state = modeler_state['World State']
    agent = state['A']
    partner = state['P']
    robbers = [robloc for rob, robloc in state.items() if 'Robber' in rob]

    def dist(loc1, loc2): return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    closest = min(max(dist(agent, rob), dist(partner, rob)) for rob in robbers)

    return 100 - (state['Round'] + closest)

cops_and_robbers_scenario = Scenario(initial_state=initial_state, actions=actions,
                                     transition=transition, utility=utility, end=end)
