import numpy as np
from collections import deque

from pdm4ar.exercises.ex04.mdp import GridMdp, GridMdpSolver
from pdm4ar.exercises.ex04.structures import ValueFunc, Policy, Action, Cell, State
from pdm4ar.exercises_def.ex04.utils import time_function


class PolicyIteration(GridMdpSolver):
    @staticmethod
    @time_function
    def solve(grid_mdp: GridMdp) -> tuple[ValueFunc, Policy]:
        value_func = np.zeros_like(grid_mdp.grid).astype(float)
        policy = np.zeros_like(grid_mdp.grid).astype(int)
        n_rows, n_cols = grid_mdp.grid.shape
        start_state = tuple(np.argwhere(grid_mdp.grid == Cell.START)[0])
        goal_state = tuple(np.argwhere(grid_mdp.grid == Cell.GOAL)[0])

        # todo implement here

        gamma = grid_mdp.gamma
        epsilon = 1e-6
        actions = list(Action)
        actions_no_abandon = [a for a in actions if a != Action.ABANDON]

        basic_offsets = [
            (0, 0),  # self (for SWAMP staying in place)
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),  # 4-neighbors only
        ]
        extended_offsets = [
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),  # diagonals
            (2, 0),
            (-2, 0),
            (0, 2),
            (0, -2),  # distance-2
        ]

        change = {
            Action.NORTH: (-1, 0),
            Action.SOUTH: (1, 0),
            Action.WEST: (0, -1),
            Action.EAST: (0, 1),
            Action.STAY: (0, 0),
        }

        # --- Precompute neighbors for each state ---
        neighbors = {}
        for state in np.ndindex(grid_mdp.grid.shape):
            r, c = state

            neighbors_list = []
            for dr, dc in basic_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    neighbors_list.append((nr, nc))

            # Check if any immediate 4-neighbor is WONDERLAND
            # If yes, we need extended neighbors for teleportation destinations
            has_wonderland_neighbor = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    if grid_mdp.grid[nr, nc] == Cell.WONDERLAND:
                        has_wonderland_neighbor = True
                        break

            # If there's a WONDERLAND neighbor, add extended neighbors
            # (teleportation can reach diagonals and distance-2 cells)
            if has_wonderland_neighbor:
                for dr, dc in extended_offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols:
                        if (nr, nc) not in neighbors_list:
                            neighbors_list.append((nr, nc))

            # Always include start state (for ABANDON action and breakdowns)
            if start_state not in neighbors_list:
                neighbors_list.append(start_state)

            neighbors[state] = tuple(neighbors_list)

        # --- Precompute valid actions for every (non-cliff, non-wonderland) state ---
        valid_actions = {}
        for state in np.ndindex(grid_mdp.grid.shape):
            cell = grid_mdp.grid[state]
            if cell in (Cell.CLIFF, Cell.WONDERLAND):
                valid_actions[state] = ()
                continue

            r, c = state
            acts = [Action.ABANDON]
            for action in actions_no_abandon:
                if action == Action.STAY and cell != Cell.GOAL:
                    continue
                dr, dc = change[action]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                    continue
                if grid_mdp.grid[nr, nc] == Cell.CLIFF:
                    continue
                acts.append(action)
            valid_actions[state] = tuple(acts)

        # --- Precompute transitions for each (state, action) pair we actually consider ---
        transitions = {}
        for state in np.ndindex(grid_mdp.grid.shape):
            if grid_mdp.grid[state] in (Cell.CLIFF, Cell.WONDERLAND):
                continue
            for action in valid_actions[state]:
                triple_list = []
                for next_state in neighbors[state]:
                    prob = grid_mdp.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        rew = grid_mdp.stage_reward(state, action, next_state)
                        triple_list.append((next_state, prob, rew))
                transitions[(state, action)] = tuple(triple_list)

        def Q(state: State, action: Action) -> float:
            q_val = 0.0
            for next_state, prob, reward in transitions.get((state, action), ()):
                q_val += prob * (reward + gamma * value_func[next_state])
            return q_val

        """
        # Precompute P(s'|s,a) and R(s,a,s')
        P = {}
        R = {}
        for state in np.ndindex(grid_mdp.grid.shape):
            cell = grid_mdp.grid[state]

            # skip CLIFF and WONDERLAND
            if cell in (Cell.CLIFF, Cell.WONDERLAND):
                continue
            for action in actions:
                for next_state in neighbors[state]:
                    prob = grid_mdp.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        P[(state, action, next_state)] = prob
                        R[(state, action, next_state)] = grid_mdp.stage_reward(state, action, next_state)

        def Q(state: State, action: Action) -> float:
            
            #Compute Q(s,a) = Σ P(s'|s,a)[R(s,a,s') + γ V(s')]
            
            q_val = 0.0
            for next_state in neighbors[state]:
                # prob = grid_mdp.get_transition_prob(state, action, next_state)
                prob = P.get((state, action, next_state), 0.0)
                if prob == 0:
                    continue
                # reward = grid_mdp.stage_reward(state, action, next_state)
                reward = R[(state, action, next_state)]
                q_val += prob * (reward + gamma * value_func[next_state])

            return q_val
        """

        # --- Compute distance from goal for ordering ---
        dist = np.full(grid_mdp.grid.shape, np.inf)
        visited = np.zeros_like(grid_mdp.grid, dtype=bool)
        queue = deque([goal_state])
        dist[goal_state] = 0
        visited[goal_state] = True
        offsets = [(1,0), (-1,0), (0,1), (0,-1)]
        while queue:
            r, c = queue.popleft()
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols:
                    if not visited[nr, nc] and grid_mdp.grid[nr, nc] not in (Cell.CLIFF, Cell.WONDERLAND):
                        dist[nr, nc] = dist[r, c] + 1
                        visited[nr, nc] = True
                        queue.append((nr, nc))

        # Order states by descending distance from goal
        ordered_states = [s for s in np.ndindex(grid_mdp.grid.shape) if grid_mdp.grid[s] not in (Cell.CLIFF, Cell.WONDERLAND)]
        ordered_states.sort(key=lambda s: -dist[s])

        # --- Policy Iteration loop ---
        stable = False
        while not stable:
            # --- Policy Evaluation ---
            while True:
                delta = 0
                #for state in np.ndindex(grid_mdp.grid.shape):
                for state in ordered_states:
                    #if grid_mdp.grid[state] in (Cell.CLIFF, Cell.WONDERLAND):
                    #    continue
                    old_v = value_func[state]
                    a = Action(policy[state])
                    value_func[state] = Q(state, a)
                    delta = max(delta, abs(old_v - value_func[state]))
                if delta < epsilon:
                    break

            """
            # --- Policy Improvement ---
            stable = True
            for state in np.ndindex(grid_mdp.grid.shape):
                if grid_mdp.grid[state] in (Cell.CLIFF, Cell.WONDERLAND):
                    continue
                
                r, c = state
                valid_actions = [Action.ABANDON]
                for action in actions_no_abandon:
                    # Only allow STAY at goal
                    if action == Action.STAY and grid_mdp.grid[state] != Cell.GOAL:
                        continue
                    dr, dc = change[action]
                    nr, nc = r + dr, c + dc
                    # Skip out-of-bounds
                    if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                        continue
                    # Skip cliffs
                    if grid_mdp.grid[nr, nc] == Cell.CLIFF:
                        continue
                    valid_actions.append(action)

                old_action = policy[state]
                q_values = [Q(state, a) for a in valid_actions]
                best_action = valid_actions[int(np.argmax(q_values))]
                # best_action = int(np.argmax(q_values))
                policy[state] = best_action
                if old_action != best_action:
                    stable = False
            """
            
            # --- Policy Improvement ---
            stable = True
            #for state in np.ndindex(grid_mdp.grid.shape):
            for state in ordered_states:
                #if grid_mdp.grid[state] in (Cell.CLIFF, Cell.WONDERLAND):
                #    continue

                # get precomputed admissible actions
                valid_actions_for_state = valid_actions[state]

                old_action = policy[state]
                q_values = [Q(state, a) for a in valid_actions_for_state]
                best_action = valid_actions_for_state[int(np.argmax(q_values))]
                policy[state] = best_action
                if old_action != best_action:
                    stable = False

        return value_func, policy
