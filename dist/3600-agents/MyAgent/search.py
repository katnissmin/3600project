import time
from game.board import Board
from game.enums import MoveType, CARPET_POINTS_TABLE
from game.move import Move
from .heuristic import evaluate


class SearchTimeout(Exception):
    pass


def order_moves(moves):
    """
    Order moves for better alpha-beta pruning:
    carpet (by length desc) > prime > plain.
    """
    def move_priority(m):
        if m.move_type == MoveType.CARPET:
            return (0, -m.roll_length)
        elif m.move_type == MoveType.PRIME:
            return (1, 0)
        elif m.move_type == MoveType.PLAIN:
            return (2, 0)
        return (3, 0)

    return sorted(moves, key=move_priority)


def negamax(board: Board, depth: int, alpha: float, beta: float,
            time_limit_func, rat_tracker=None) -> float:
    """
    Negamax search with alpha-beta pruning.

    Uses reverse_perspective after each move, so evaluate() always returns
    the score from the current player_worker's view. Negation handles the
    adversarial alternation correctly.
    """
    if time_limit_func() < 0.05:
        raise SearchTimeout()

    if depth == 0 or board.is_game_over():
        return evaluate(board, rat_tracker)

    moves = board.get_valid_moves(enemy=False, exclude_search=True)
    if not moves:
        return evaluate(board, rat_tracker)

    moves = order_moves(moves)

    best_val = float('-inf')
    for move in moves:
        child = board.forecast_move(move, check_ok=False)
        if child is None:
            continue
        child.reverse_perspective()

        val = -negamax(child, depth - 1, -beta, -alpha, time_limit_func, rat_tracker)
        if val > best_val:
            best_val = val
        alpha = max(alpha, val)
        if alpha >= beta:
            break

    return best_val


def find_best_move(board: Board, time_budget: float, time_left_func,
                   rat_tracker=None):
    """
    Iterative deepening negamax. Returns (best_move, best_score).
    """
    start_time = time.perf_counter()

    def move_time_left():
        elapsed = time.perf_counter() - start_time
        return min(time_budget - elapsed, time_left_func() - 0.5)

    moves = board.get_valid_moves(enemy=False, exclude_search=True)
    if not moves:
        return None, float('-inf')

    moves = order_moves(moves)
    best_move = moves[0]
    best_score = float('-inf')

    for depth in range(1, 20):
        if move_time_left() < 0.1:
            break

        try:
            current_best_move = None
            current_best_score = float('-inf')
            alpha = float('-inf')
            beta = float('inf')

            for move in moves:
                child = board.forecast_move(move, check_ok=False)
                if child is None:
                    continue
                child.reverse_perspective()

                score = -negamax(child, depth - 1, -beta, -alpha,
                                 move_time_left, rat_tracker)

                if score > current_best_score:
                    current_best_score = score
                    current_best_move = move
                alpha = max(alpha, score)

            if current_best_move is not None:
                best_move = current_best_move
                best_score = current_best_score

                # reorder moves: put best move first for next iteration
                moves = [best_move] + [m for m in moves if m is not best_move]

        except SearchTimeout:
            break

    return best_move, best_score
