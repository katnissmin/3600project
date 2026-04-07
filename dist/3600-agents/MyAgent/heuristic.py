from game.enums import Cell, BOARD_SIZE, CARPET_POINTS_TABLE, Direction, MoveType, MAX_TURNS_PER_PLAYER
from game.board import Board

NUM_CELLS = BOARD_SIZE * BOARD_SIZE

W_SCORE_DIFF = 1.0
W_PRIMED_COUNT = 0.25
W_CARPET_POTENTIAL = 0.6
W_LINE_EXTEND = 0.15
W_POSITION = 0.1
W_OPPONENT_THREAT = 0.35
W_RAT_EV = 0.35
W_SPACE_ACCESS = 0.08


def evaluate(board: Board, rat_tracker=None) -> float:
    """
    Evaluate a board state from the current player's perspective.
    Higher is better for the current player.
    """
    my_worker = board.player_worker
    opp_worker = board.opponent_worker
    my_pos = my_worker.get_location()
    opp_pos = opp_worker.get_location()
    turns_left = my_worker.turns_left

    score = 0.0

    score += W_SCORE_DIFF * (my_worker.get_points() - opp_worker.get_points())

    primed_count = _count_primed(board)
    my_carpet_val, my_best_len = _carpet_potential(board, my_pos)
    opp_carpet_val, _ = _carpet_potential(board, opp_pos)

    phase = _game_phase(turns_left)

    if phase == 0:  # early
        score += W_PRIMED_COUNT * primed_count * 1.2
        score += W_CARPET_POTENTIAL * my_carpet_val * 0.5
        score += W_SPACE_ACCESS * _space_accessibility(board, my_pos) * 1.5
        score += W_LINE_EXTEND * _line_extendability(board, my_pos)
    elif phase == 1:  # mid
        score += W_PRIMED_COUNT * primed_count
        score += W_CARPET_POTENTIAL * my_carpet_val
        score += W_LINE_EXTEND * _line_extendability(board, my_pos)
        score += W_SPACE_ACCESS * _space_accessibility(board, my_pos)
    else:  # late
        score += W_PRIMED_COUNT * primed_count * 0.3
        score += W_CARPET_POTENTIAL * my_carpet_val * 1.5
        score += W_SPACE_ACCESS * _space_accessibility(board, my_pos) * 0.3

    score -= W_OPPONENT_THREAT * opp_carpet_val

    score += W_POSITION * _positional_value(board, my_pos)

    if rat_tracker is not None:
        _, _, best_ev = rat_tracker.best_search_cell()
        if best_ev > 0:
            score += W_RAT_EV * best_ev

    return score


def _game_phase(turns_left):
    """0 = early (>25 turns), 1 = mid (10-25 turns), 2 = late (<10 turns)."""
    if turns_left > 25:
        return 0
    elif turns_left > 10:
        return 1
    return 2


def _count_primed(board: Board) -> int:
    return bin(board._primed_mask).count('1')


def _carpet_potential(board: Board, worker_pos):
    """
    Evaluate carpet scoring potential near the worker.
    Returns (best_carpet_value, best_line_length).
    """
    wx, wy = worker_pos
    best_value = 0.0
    best_len = 0

    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for dx, dy in directions:
        length = 0
        cx, cy = wx + dx, wy + dy
        while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
            bit = 1 << (cy * BOARD_SIZE + cx)
            if board._primed_mask & bit:
                length += 1
                cx += dx
                cy += dy
            else:
                break

        if length > 0 and length in CARPET_POINTS_TABLE:
            value = CARPET_POINTS_TABLE[length]
            if value > best_value:
                best_value = value
                best_len = length

    return best_value, best_len


def _line_extendability(board: Board, worker_pos) -> float:
    """
    Score how easily the worker can extend primed lines.
    Reward being on a space cell adjacent to primed lines (can prime and extend).
    """
    wx, wy = worker_pos
    my_bit = 1 << (wy * BOARD_SIZE + wx)

    on_space = bool(board._space_mask & my_bit) and not bool(
        (board._primed_mask | board._carpet_mask | board._blocked_mask) & my_bit
    )
    if not on_space:
        return 0.0

    score = 0.0
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for dx, dy in directions:
        nx, ny = wx + dx, wy + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            bit = 1 << (ny * BOARD_SIZE + nx)
            if board._primed_mask & bit:
                length = 1
                cx, cy = nx + dx, ny + dy
                while 0 <= cx < BOARD_SIZE and 0 <= cy < BOARD_SIZE:
                    b = 1 << (cy * BOARD_SIZE + cx)
                    if board._primed_mask & b:
                        length += 1
                        cx += dx
                        cy += dy
                    else:
                        break
                if length + 1 in CARPET_POINTS_TABLE:
                    gain = CARPET_POINTS_TABLE[length + 1] - CARPET_POINTS_TABLE.get(length, 0)
                    score += max(gain, 0)

    return score


def _space_accessibility(board: Board, worker_pos) -> float:
    """Count accessible open space cells within 3 Manhattan distance."""
    wx, wy = worker_pos
    count = 0
    blocked = board._blocked_mask | board._primed_mask | board._carpet_mask
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            dist = abs(dx) + abs(dy)
            if dist == 0 or dist > 3:
                continue
            nx, ny = wx + dx, wy + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                bit = 1 << (ny * BOARD_SIZE + nx)
                if not (blocked & bit):
                    count += 1
    return float(count)


def _positional_value(board: Board, my_pos) -> float:
    """
    Positional advantage: prefer being near primeable space and primed lines,
    penalize being in corners or near board edges.
    """
    mx, my = my_pos
    value = 0.0

    center_x = (BOARD_SIZE - 1) / 2.0
    center_y = (BOARD_SIZE - 1) / 2.0
    dist_to_center = abs(mx - center_x) + abs(my - center_y)
    value -= dist_to_center * 0.1

    for dx in range(-2, 3):
        for dy in range(-2, 3):
            nx, ny = mx + dx, my + dy
            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                bit = 1 << (ny * BOARD_SIZE + nx)
                dist = abs(dx) + abs(dy)
                if dist == 0:
                    continue
                weight = 1.0 / dist
                if board._primed_mask & bit:
                    value += weight * 1.0
                elif board._space_mask & bit and not (board._blocked_mask & bit):
                    value += weight * 0.3

    return value
