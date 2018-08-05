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
from collections import namedtuple

from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace

WALL, OPEN, AGENT, PARTNER, ROBBER, GATE_UP, GATE_DOWN, GATE_RIGHT, GATE_LEFT = \
    '*', ' ', 'A', 'S', 'R', '^', 'v', '>', '<'

Location = namedtuple('Location', ['row', 'col'])
_path = './domains/multi_agent/cops_and_robbers/mazes/'


class CopsAndRobbersScenario:

    def __init__(self, filename='a.maze'):
        """
        Open the file and read in the maze configuration.
        """
        with open(_path + filename, 'r') as maze_file:
            maze_lines = maze_file.read().split('\n')

        self.initial_maze = {Location(row, col): char for row, line in enumerate(maze_lines) for col, char in enumerate(line)}
        replace = set((AGENT, PARTNER, ROBBER))
        self.maze = {loc : char if char not in replace else OPEN for loc, char in self.initial_maze.items()}

    def agents(self):
        return ['A','P']

    def heuristic(self, state):
        return 0

    def initial_state(self):
        """
        Returns the initial state, as constructed from the maze file. Notably, this sets the locations of the agents.
        """
        state_dict = {'Round': 1}

        robber_count = 0
        for loc, char in self.initial_maze.items():

            if char == AGENT:
                state_dict['A'] = loc

            elif char == PARTNER:
                state_dict['P'] = loc

            elif char == ROBBER:
                robber_count += 1
                state_dict['Robber' + str(robber_count)] = loc

        assert len(state_dict) > 2, 'Improper maze comprehension. State=' + str(state_dict)
        return State(state_dict)

    def actions(self, state):
        """
        Returns the legal actions available to all agents.
        """
        action_lists = {}
        for agent_name in ['A', 'P']:
            row, col = state[agent_name]

            legal_actions = ['W']
            if self.maze[(row + 1, col)] == OPEN or self.maze[(row + 1, col)] == GATE_DOWN:
                legal_actions.append('D')
            if self.maze[(row - 1, col)] == OPEN or self.maze[(row - 1, col)] == GATE_UP:
                legal_actions.append('U')
            if self.maze[(row, col + 1)] == OPEN or self.maze[(row, col + 1)] == GATE_RIGHT:
                legal_actions.append('R')
            if self.maze[(row, col - 1)] == OPEN or self.maze[(row, col - 1)] == GATE_LEFT:
                legal_actions.append('L')

            action_lists[agent_name] = legal_actions

        return JointActionSpace(action_lists)

    def transition(self, state, action):
        """
        In order for the end check to work (both agents and one robber in the same cell), it is necessary to move the
        robbers before the agents. Otherwise, the robber will always slip away.
        """
        intermediate_state_distribution = self._move_robbers(state)

        new_state_distribution = Distribution()
        for intermediate_state, probability in intermediate_state_distribution.items():
            new_state_diff = {'Round': intermediate_state['Round'] + 1}

            for agent, individual_action in action.items():
                row, col = intermediate_state[agent]
                if individual_action == 'D':
                    row += 1
                elif individual_action == 'U':
                    row -= 1
                elif individual_action == 'R':
                    col += 1
                elif individual_action == 'L':
                    col -= 1

                assert 0 < row < 8 and 0 < col < 8, \
                    'Illegal action taken. {action} {loc}'.format(action=individual_action, loc=(row, col)) + '\n' + \
                    self.show_state(intermediate_state)

                new_state_diff[agent] = Location(row, col)

            new_state = intermediate_state.update(new_state_diff)
            assert len(new_state) > 2, 'Improper state update. ' + str(new_state)
            new_state_distribution[new_state] = probability

        return new_state_distribution

    def end(self, state):
        """
        End conditions:
            - Round limit hit. Currently 50.
            - Both agents and at least one robber are located in a single cell.
        """
        if state['Round'] >= 13:
            return True

        a_loc, p_loc = state['A'], state['P']
        for loc in [loc for key, loc in state.items() if 'Robber' in key]:
            if a_loc == p_loc == loc:
                return True

        return False

    def utility(self, old_state, action, new_state):
        """
        Utility is only granted upon successful completion of the task. It is given as the number of remaining rounds.
        """
        return (50 - new_state['Round']) if self.end(new_state) else 0

    def _move_robbers(self, state):
        """
        Robbers move to the nearest adjacent cell which maximizes the minimum distance to the two pursuing agents.
        If a tie exists, it is broken by distance to just the main agent. If the tie persists, all tiles are given
        equal probability.
        """
        resulting_distribution = Distribution({state: 1.0})

        agent_loc, partner_loc = state['A'], state['P']
        for robber, loc in [(key, loc) for key, loc in state.items() if key.startswith('Robber')]:
            row, col = loc

            # Identify all legal targets for moving to.
            targets = [loc]
            if self.maze[(row + 1, col)] == OPEN or self.maze[(row + 1, col)] == GATE_DOWN:
                targets.append(Location(row + 1, col))
            if self.maze[(row - 1, col)] == OPEN or self.maze[(row - 1, col)] == GATE_UP:
                targets.append(Location(row - 1, col))
            if self.maze[(row, col + 1)] == OPEN or self.maze[(row, col + 1)] == GATE_RIGHT:
                targets.append(Location(row, col + 1))
            if self.maze[(row, col - 1)] == OPEN or self.maze[(row, col - 1)] == GATE_LEFT:
                targets.append(Location(row, col - 1))
            shuffle(targets)

            assert all(0 < row < 8 and 0 < col < 8 for row, col in targets), 'Illegal robber target. ' + '\n'.join(str(target) for target in targets)

            # Use Manhattan distance for convenience.
            def distance(p1, p2): return abs(p1.row - p2.row) + abs(p1.col - p2.col)
            distances = [(cell, distance(cell, agent_loc), distance(cell, partner_loc)) for cell in targets]

            # Find the best target maximizing the minimum distance to the players.
            best_target = max(distances, key=lambda x: min(x[1], x[2]))
            ties = [d for d in distances if min(d[1], d[2]) == min(best_target[1], best_target[2])]

            # Break ties on distance to just agent
            best_target = max(ties, key=lambda x: x[1])
            ties = [d for d in ties if d[1] == best_target[1]]

            # Update state distribution.
            new_state_dist = Distribution()
            for result_state, state_prob in resulting_distribution.items():
                tie_prob = 1/len(ties)
                for target, _, _ in ties:
                    new_state = result_state.update({robber: target})  # state.update() returns a modified copy
                    new_state_dist[new_state] = tie_prob * state_prob
            resulting_distribution = new_state_dist

        # Safety check to ensure we haven't calculated new state probabilities incorrectly.
        assert abs(sum(resulting_distribution.values()) - 1.0) < 10e-5, 'Resulting transition too large. ' + str(sum(resulting_distribution.values()))

        for state in resulting_distribution:
            for robber_loc in [val for key, val in state.items() if 'Robber' in key]:
                row, col = robber_loc
                assert 0 < row < 8 and 0 < col < 8, 'Illegal robber location. Bad move. ' + str((row, col))

        return resulting_distribution

    def show_state(self, state):
        """
        Returns a printable string representation of the state, showing the locations of the agents within the maze.
        """
        robbers = [key for key in state if 'Robber' in key]
        players = [('A', 'A'), ('P', 'P')] + [(rob, 'R') for rob in robbers]

        maze_copy = self.maze.copy()
        for name, char in players:
            maze_copy[state[name]] = char

        string = ''
        for row in range(1 + max(k[0] for k in maze_copy.keys())):
            for col in range(1 + max(k[1] for k in maze_copy.keys())):
                string += maze_copy[Location(row, col)]
            string += '\n'

        string += 'Agent: {agent} Partner: {partner} Round: {round}\n'.format(agent=state['A'],
                                                                              partner=state['P'],
                                                                              round=state['Round'])
        return string


def base_heuristic(state):
    """
    A simple non-admissable heuristic: find the robber who has the smallest distance to the farthest agent.
    Estimate: 50 (max rounds) - current round - the distance
    """
    agent, partner = state['A'], state['P']
    robbers = [robloc for rob, robloc in state.items() if 'Robber' in rob]

    # Use Manhattan distance to give a sense of how quickly the agents may capture the nearest robber.
    def dist(loc1, loc2): return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])
    closest = min(max(dist(agent, rob), dist(partner, rob)) for rob in robbers)

    return 50 - (state['Round'] + closest)


def modeling_heuristic(modeler_state):
    """
    As the modeling agent has extra state information, this heuristic simply calls the base heuristic
    function ont he appropriate world-only state representation.
    """
    return base_heuristic(modeler_state['World State'])


