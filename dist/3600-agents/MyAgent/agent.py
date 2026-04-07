from collections.abc import Callable
from typing import Tuple

import numpy as np

from game import board, move, enums
from game.board import Board
from game.move import Move

from .rat_tracker import RatTracker
from .strategy import Strategy


class PlayerAgent:
    """
    Competitive agent using expectiminimax search with HMM-based rat tracking.
    """

    def __init__(self, board: Board, transition_matrix=None, time_left: Callable = None):
        T = np.array(transition_matrix, dtype=np.float64)
        self.rat_tracker = RatTracker(T)
        self.strategy = Strategy(self.rat_tracker)
        self.turns_played = 0

    def commentate(self):
        catches = self.strategy.total_rat_catches
        turns = self.turns_played
        return f"Played {turns} turns, caught rat {catches} time(s)."

    def play(
        self,
        board: Board,
        sensor_data: Tuple,
        time_left: Callable,
    ) -> Move:
        self.turns_played += 1
        return self.strategy.decide(board, sensor_data, time_left)
