from abc import ABC, abstractmethod
from pickle import TRUE
from types import CellType
import math

from networkx import neighbors
import numpy as np
from numpy.typing import NDArray
from pdm4ar.exercises.ex04.structures import Action, Policy, State, ValueFunc, Cell


class GridMdp:
    def __init__(self, grid: NDArray[np.int64], gamma: float = 0.9):
        assert len(grid.shape) == 2, "Map is invalid"
        self.grid = grid
        """The map"""
        self.gamma: float = gamma
        """Discount factor"""

    def get_transition_prob(self, state: State, action: Action, next_state: State) -> float:
        """Returns P(next_state | state, action)"""
        # todo

        def get_cell_safe(row, col):
            if 0 <= row < n_rows and 0 <= col < n_cols:
                return Cell(self.grid[row, col])
            return Cell.CLIFF

        n_rows, n_cols = self.grid.shape
        r, c = state

        cell_type = Cell(self.grid[state])
        next_cell_type = Cell(self.grid[next_state])

        # no movement stuff action
        if action == Action.ABANDON:
            if next_cell_type == Cell.START:
                return 1.0
            else:
                return 0.0

        if action == Action.STAY:
            if cell_type == Cell.GOAL and next_cell_type == Cell.GOAL:
                return 1.0
            else:
                return 0.0

        intended_state = {
            Action.NORTH: (r - 1, c),
            Action.SOUTH: (r + 1, c),
            Action.WEST: (r, c - 1),
            Action.EAST: (r, c + 1),
        }

        intendend_cell_types = {
            Action.NORTH: get_cell_safe(r - 1, c),
            Action.SOUTH: get_cell_safe(r + 1, c),
            Action.WEST: get_cell_safe(r, c - 1),
            Action.EAST: get_cell_safe(r, c + 1),
        }

        actions = {
            Action.NORTH: [(0, -1), (-1, 0), (0, 1)],
            Action.SOUTH: [(0, -1), (1, 0), (0, 1)],
            Action.WEST: [(0, -1), (1, 0), (-1, 0)],
            Action.EAST: [(0, 1), (1, 0), (-1, 0)],
        }

        opposite_action = {
            Action.NORTH: get_cell_safe(1, 0),
            Action.SOUTH: get_cell_safe(-1, 0),
            Action.WEST: get_cell_safe(0, 1),
            Action.EAST: get_cell_safe(0, -1),
        }

        # VALID ACTION TEST: test if action goes into CLIFF or out of bounce
        if intendend_cell_types[action] == Cell.CLIFF:
            return 0.0

        # Probability to be in a cliff or wonderland is 0
        if cell_type == Cell.WONDERLAND or cell_type == Cell.CLIFF:
            return 0.0

        # Probability to land in a cliff or wonderland is 0
        if next_cell_type == Cell.WONDERLAND or next_cell_type == Cell.CLIFF:
            return 0.0

        if cell_type == Cell.START:
            if next_state == intended_state[action]:
                return 0.75
            if next_state in intended_state.values():
                return 0.25 / 3.0

        if cell_type == Cell.GRASS:
            breakdown_prob = 0
            if next_state == intended_state[action]:
                return 0.75
            if next_state in intended_state.values():
                return 0.25 / 3.0

            for neighbor in intendend_cell_types.values():
                if neighbor == Cell.CLIFF:
                    breakdown_prob += 0.25 / 3.0

            if intendend_cell_types[action] == Cell.WONDERLAND:
                WL_r, WL_c = intended_state[action]

                teleport_targets = []
                for dx, dy in actions[action]:
                    tr, tc = WL_r + dx, WL_c + dy
                    teleport_targets.append((tr, tc))

                for tr, tc in teleport_targets:
                    if get_cell_safe(tr, tc) == Cell.CLIFF:
                        breakdown_prob += 0.75 / 3.0
                    if (tr, tc) == next_state:
                        return 0.75 / 3.0

            other_actions = [a for a in [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST] if a != action]
            for slip_action in other_actions:
                if intendend_cell_types[slip_action] == Cell.WONDERLAND:
                    WL_r, WL_c = intended_state[slip_action]

                    teleport_targets = []
                    for dx, dy in actions[slip_action]:
                        tr, tc = WL_r + dx, WL_c + dy
                        teleport_targets.append((tr, tc))

                    for tr, tc in teleport_targets:
                        if get_cell_safe(tr, tc) == Cell.CLIFF:
                            breakdown_prob += 0.25 / 3.0 / 3.0
                        if (tr, tc) == next_state:
                            return 0.25 / 3.0 / 3.0

            if next_cell_type == Cell.START:
                return breakdown_prob

        if cell_type == Cell.SWAMP:
            breakdown_prob = 0.05
            if next_state == intended_state[action]:
                return 0.5
            if next_state in intended_state.values():
                return 0.25 / 3.0
            if next_state == state:
                return 0.2

            for neighbor in intendend_cell_types.values():
                if neighbor == Cell.CLIFF:
                    breakdown_prob += 0.25 / 3.0

            if intendend_cell_types[action] == Cell.WONDERLAND:
                WL_r, WL_c = intended_state[action]

                teleport_targets = []
                for dx, dy in actions[action]:
                    tr, tc = WL_r + dx, WL_c + dy
                    teleport_targets.append((tr, tc))

                for tr, tc in teleport_targets:
                    if get_cell_safe(tr, tc) == Cell.CLIFF:
                        breakdown_prob += 0.5 / 3.0
                    if (tr, tc) == next_state:
                        return 0.5 / 3.0

            other_actions = [a for a in [Action.NORTH, Action.SOUTH, Action.EAST, Action.WEST] if a != action]
            for slip_action in other_actions:
                if intendend_cell_types[slip_action] == Cell.WONDERLAND:
                    WL_r, WL_c = intended_state[slip_action]

                    teleport_targets = []
                    for dx, dy in actions[slip_action]:
                        tr, tc = WL_r + dx, WL_c + dy
                        teleport_targets.append((tr, tc))

                    for tr, tc in teleport_targets:
                        if get_cell_safe(tr, tc) == Cell.CLIFF:
                            breakdown_prob += 0.25 / 3.0 / 3.0
                        elif (tr, tc) == next_state:
                            return 0.25 / 3.0 / 3.0

            if next_cell_type == Cell.START:
                return breakdown_prob

        return 0.0

    def stage_reward(self, state: State, action: Action, next_state: State) -> float:
        # todo

        cell_type = Cell(self.grid[state])
        next_cell_type = Cell(self.grid[next_state])

        if action == Action.ABANDON:
            return -10

        if cell_type == Cell.GOAL and action == Action.STAY and next_state == state:
            return 50

        if cell_type == Cell.START:
            return -1

        distance = np.linalg.norm(np.array(state) - np.array(next_state))

        if next_cell_type == Cell.START and distance > 1:
            if cell_type == Cell.SWAMP:
                return -10 - 2
            if cell_type == Cell.GRASS:
                return -10 - 1

        if cell_type == Cell.GRASS:
            if distance == 1:
                return -1
            if 1 < distance <= 2:
                return -1 + 3

        if cell_type == Cell.SWAMP:
            if distance == 1 or distance == 0:
                return -2
            if 1 < distance <= 2:
                return -2 + 3

        return 0


class GridMdpSolver(ABC):
    @staticmethod
    @abstractmethod
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        pass
