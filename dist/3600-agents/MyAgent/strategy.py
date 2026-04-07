from game.board import Board
from game.move import Move
from game.enums import MAX_TURNS_PER_PLAYER, CARPET_POINTS_TABLE

from .rat_tracker import RatTracker
from .search import find_best_move

RAT_SEARCH_EV_THRESHOLD = 0.5

TIME_FLOOR = 0.3
TIME_CEIL = 6.0
TIME_SAFETY_MARGIN = 5.0


class Strategy:
    """High-level decision-making: board move vs. rat search, time allocation."""

    def __init__(self, rat_tracker: RatTracker):
        self.rat_tracker = rat_tracker
        self.turn_number = 0
        self.total_rat_catches = 0

    def decide(self, board: Board, sensor_data, time_left_func):
        """
        Main entry: decide what move to make this turn.

        Args:
            board: current Board state
            sensor_data: (noise, estimated_distance) tuple
            time_left_func: callable returning seconds remaining

        Returns:
            Move object
        """
        noise, estimated_distance = sensor_data
        self.turn_number += 1

        self.rat_tracker.predict()
        worker_pos = board.player_worker.get_location()
        self.rat_tracker.update(noise, estimated_distance, worker_pos, board)

        self._handle_opponent_search(board)

        time_budget = self._allocate_time(board, time_left_func)

        _, best_search_pos, best_search_ev = self.rat_tracker.best_search_cell()

        board_move, board_score = find_best_move(
            board, time_budget * 0.85, time_left_func, self.rat_tracker
        )

        if self._should_search_rat(best_search_ev, board, board_move, board_score):
            return Move.search(best_search_pos)

        if board_move is not None:
            return board_move

        moves = board.get_valid_moves(enemy=False, exclude_search=True)
        if moves:
            return moves[0]

        return Move.search(best_search_pos)

    def _should_search_rat(self, search_ev, board, board_move, board_score) -> bool:
        """Decide if searching for the rat is better than making a board move."""
        if search_ev < RAT_SEARCH_EV_THRESHOLD:
            return False

        turns_left = board.player_worker.turns_left

        if turns_left <= 3 and search_ev > 1.0:
            return True

        if search_ev > 2.0:
            return True

        if self.rat_tracker.max_belief() > 0.5:
            return True

        return False

    def _handle_opponent_search(self, board: Board):
        """If opponent or we caught the rat, reset tracker."""
        opp_search_loc, opp_search_result = board.opponent_search
        if opp_search_result is True:
            self.rat_tracker.reset_on_capture()
            self.rat_tracker.predict()

        my_search_loc, my_search_result = board.player_search
        if my_search_result is True:
            self.total_rat_catches += 1
            self.rat_tracker.reset_on_capture()
            self.rat_tracker.predict()

    def _allocate_time(self, board: Board, time_left_func) -> float:
        """Allocate time budget for this move."""
        remaining_time = time_left_func()
        turns_left = max(board.player_worker.turns_left, 1)

        budget = (remaining_time - TIME_SAFETY_MARGIN) / turns_left

        budget = max(budget, TIME_FLOOR)
        budget = min(budget, TIME_CEIL)
        budget = min(budget, remaining_time - 1.0)

        return max(budget, 0.1)

    def notify_rat_caught(self):
        """Call when our search successfully caught the rat."""
        self.total_rat_catches += 1
        self.rat_tracker.reset_on_capture()
