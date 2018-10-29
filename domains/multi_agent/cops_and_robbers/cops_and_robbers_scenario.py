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
from collections import namedtuple

from mdp.distribution import Distribution
from mdp.state import State
from mdp.action import JointActionSpace

import json
import os

WALL, OPEN, AGENT, PARTNER, ROBBER, GATE_UP, GATE_DOWN, GATE_RIGHT, GATE_LEFT = \
    '*', ' ', 'A', 'S', 'R', '^', 'v', '>', '<'

Location = namedtuple('Location', ['row', 'col'])
_path = './domains/multi_agent/cops_and_robbers/mazes/'


class CopsAndRobbersScenario:

    def __init__(self, filename='simple.maze', max_rounds=6, reward=100):
        """
        Open the file and read in the maze configuration.
        """
        with open(_path + filename, 'r') as maze_file:
            maze_lines = maze_file.read().split('\n')

        self.initial_maze = {Location(row, col): char for row, line in enumerate(maze_lines) for col, char in enumerate(line)}
        replace = {AGENT, PARTNER, ROBBER}
        self.maze = {loc: char if char not in replace else OPEN for loc, char in self.initial_maze.items()}

        self._state_transition_cache = {}
        self.max_rounds = max_rounds
        self.success_reward = reward

    def agents(self):
        return ['A', 'P']

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
        if (state, action) in self._state_transition_cache:
            return self._state_transition_cache[(state, action)]

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
            new_state_distribution[new_state] = probability

        self._state_transition_cache[(state, action)] = new_state_distribution

        return new_state_distribution

    def end(self, state):
        return self.end_check(state)

    def end_check(self, state):
        """
        End conditions:
            - Round limit hit. Currently 50.
            - Both agents and at least one robber are located in a single cell.
        """
        if state['Round'] > self.max_rounds:
            return True

        return self.robber_caught(state)

    def robber_caught(self, state):
        a_loc, p_loc = state['A'], state['P']
        for loc in [loc for key, loc in state.items() if 'Robber' in key]:
            if a_loc == p_loc == loc:
                return True

        return False

    def utility(self, old_state, action, new_state):
        """
        Utility is only granted upon successful completion of the task. It is given as the number of remaining rounds.
        """
        return self.success_reward if self.robber_caught(new_state) else 0

    def _serialize_state(self, state):
        return json.dumps({k: tuple(v) if type(v) is Location else v for k, v in state.items()})

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
            tie_prob = 1/len(ties)
            for result_state, state_prob in resulting_distribution.items():
                for target, _, _ in ties:
                    new_state = result_state.update_item(robber, target)  # state.update() returns a modified copy
                    new_state_dist[new_state] = tie_prob * state_prob
            resulting_distribution = new_state_dist

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

    def heuristic(self, state):
        """ If any robber is within (Manhattan) distance of both cops, return an optimistic reward. """
        if self.end(state):
            return self.success_reward if self.robber_caught(state) else 0

        agent_loc = state['A']
        partner_loc = state['P']
        rounds_left = self.max_rounds - state['Round']

        if any(self.distance(agent_loc, r_loc) <= rounds_left and self.distance(partner_loc, r_loc) <= rounds_left
               for r, r_loc in state.items() if r.startswith('Rob')):
            return self.success_reward

        return 0

    def distance(self, loc1, loc2):
        return abs(loc1.col - loc2.col) + abs(loc1.row - loc2.row)



