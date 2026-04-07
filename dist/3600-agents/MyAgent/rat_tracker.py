import numpy as np
from game.enums import Cell, Noise, BOARD_SIZE

NOISE_PROBS = {
    Cell.BLOCKED: (0.5, 0.3, 0.2),
    Cell.SPACE:   (0.7, 0.15, 0.15),
    Cell.PRIMED:  (0.1, 0.8, 0.1),
    Cell.CARPET:  (0.1, 0.1, 0.8),
}

DISTANCE_ERROR_PROBS = np.array([0.12, 0.70, 0.12, 0.06])
DISTANCE_ERROR_OFFSETS = np.array([-1, 0, 1, 2])

NUM_CELLS = BOARD_SIZE * BOARD_SIZE


class RatTracker:
    """Hidden Markov Model for tracking the rat's position."""

    def __init__(self, transition_matrix):
        """
        Args:
            transition_matrix: 64x64 numpy array. T[i, j] = P(move from i to j).
        """
        self.T = np.array(transition_matrix, dtype=np.float64)
        self.belief = np.ones(NUM_CELLS, dtype=np.float64) / NUM_CELLS

        self._precompute_respawn_belief()
        self._precompute_noise_table()
        self._precompute_distance_tables()

    def _precompute_respawn_belief(self):
        """Precompute belief after rat spawns at (0,0) and takes 1000 steps.
        Uses repeated squaring via matrix_power which is O(log n) multiplications."""
        start = np.zeros(NUM_CELLS, dtype=np.float64)
        start[0] = 1.0  # rat starts at (0,0), index = 0*8+0 = 0
        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.respawn_belief = start @ T_1000
        self.respawn_belief /= self.respawn_belief.sum()

    def _precompute_noise_table(self):
        """Build a (4, 3) table: noise_table[cell_type][noise_type] = P(noise | cell)."""
        self.noise_table = np.zeros((4, 3), dtype=np.float64)
        for cell_type, probs in NOISE_PROBS.items():
            self.noise_table[int(cell_type)] = probs

    def _precompute_distance_tables(self):
        """Precompute Manhattan distances from every cell to every other cell."""
        coords = np.array([(i % BOARD_SIZE, i // BOARD_SIZE) for i in range(NUM_CELLS)])
        self.coords = coords
        self.manhattan = np.zeros((NUM_CELLS, NUM_CELLS), dtype=np.int32)
        for i in range(NUM_CELLS):
            self.manhattan[i] = np.abs(coords[i, 0] - coords[:, 0]) + np.abs(coords[i, 1] - coords[:, 1])

    def predict(self):
        """Prediction step: propagate belief through transition matrix."""
        self.belief = self.belief @ self.T
        self.belief /= self.belief.sum()

    def update(self, noise, estimated_distance, worker_pos, board):
        """
        Observation update step.

        Args:
            noise: Noise enum (SQUEAK=0, SCRATCH=1, SQUEAL=2)
            estimated_distance: noisy Manhattan distance estimate (int >= 0)
            worker_pos: (x, y) tuple of current worker position
            board: Board object to query cell types
        """
        noise_idx = int(noise)
        worker_flat = worker_pos[1] * BOARD_SIZE + worker_pos[0]

        cell_types = self._get_cell_type_array(board)
        noise_likelihood = self.noise_table[cell_types, noise_idx]

        true_distances = self.manhattan[worker_flat]
        dist_likelihood = self._distance_likelihood(estimated_distance, true_distances)

        self.belief *= noise_likelihood * dist_likelihood

        total = self.belief.sum()
        if total > 0:
            self.belief /= total
        else:
            self.belief = np.ones(NUM_CELLS, dtype=np.float64) / NUM_CELLS

    def _get_cell_type_array(self, board):
        """Extract cell types for all 64 cells as a numpy array using bitwise ops."""
        cell_types = np.full(NUM_CELLS, int(Cell.SPACE), dtype=np.int32)
        for i in range(NUM_CELLS):
            bit = 1 << i
            if board._blocked_mask & bit:
                cell_types[i] = int(Cell.BLOCKED)
            elif board._primed_mask & bit:
                cell_types[i] = int(Cell.PRIMED)
            elif board._carpet_mask & bit:
                cell_types[i] = int(Cell.CARPET)
        return cell_types

    def _distance_likelihood(self, observed, true_distances):
        """
        Compute P(observed_distance | true_distance) for all cells.

        The observed distance = true_distance + offset, where offset in {-1,0,+1,+2}.
        So true_distance = observed - offset.
        But we also clamp: if true + offset < 0, observed = 0.
        """
        likelihood = np.zeros(NUM_CELLS, dtype=np.float64)
        for k, (offset, prob) in enumerate(zip(DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS)):
            if observed == 0:
                mask = (true_distances + offset) <= 0
            else:
                mask = (true_distances + offset) == observed
            likelihood[mask] += prob
        return likelihood

    def reset_on_capture(self):
        """Reset belief after rat is caught (respawns at (0,0) with 1000 moves)."""
        self.belief = self.respawn_belief.copy()

    def best_search_cell(self):
        """
        Return (flat_index, (x, y), expected_value) for the highest-EV search cell.

        EV(search at cell) = belief[cell] * 4 - (1 - belief[cell]) * 2
                           = 6 * belief[cell] - 2
        """
        ev = 6.0 * self.belief - 2.0
        best_idx = np.argmax(ev)
        best_ev = ev[best_idx]
        best_pos = (int(best_idx % BOARD_SIZE), int(best_idx // BOARD_SIZE))
        return best_idx, best_pos, best_ev

    def max_belief(self):
        """Return the maximum belief probability."""
        return float(np.max(self.belief))

    def get_belief_at(self, pos):
        """Get belief probability at a given (x, y) position."""
        flat = pos[1] * BOARD_SIZE + pos[0]
        return float(self.belief[flat])
