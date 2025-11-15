from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.polyglot as polyglot
import random
import time

# ============================================================
# GLOBALS
# ============================================================

# Transposition table:
# key -> (depth, flag, score, best_move)
TT = {}

INFINITY = 10_000_000
MAX_DEPTH = 3  # you can bump this to 4 later if it's fast enough

# TT flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# ------------------------------------------------------------
# EMBEDDED OPENING BOOK (based on UCI move sequence)
# ------------------------------------------------------------
# Key: tuple of UCI moves from the start of the game
# Value: next move in UCI for our side
#
# Very small book, just to give sane openings:
# - Ruy Lopez (1. e4 e5 2. Nf3 Nc6 3. Bb5)
# - Italian (1. e4 e5 2. Nf3 Nc6 3. Bc4)
# - Sicilian (1. e4 c5 2. Nf3 d6 3. d4)
# - Queen's Gambit (1. d4 d5 2. c4)
# - Simple London-ish (1. d4 Nf6 2. Nf3 d5 3. Bf4)
OPENING_BOOK = {
    # --- e4 e5 Ruy Lopez / Italian skeleton ---
    (): "e2e4",  # 1. e4
    ("e2e4",): "e7e5",  # ... e5
    ("e2e4", "e7e5"): "g1f3",  # 2. Nf3
    ("e2e4", "e7e5", "g1f3"): "b8c6",  # ... Nc6

    # From here, pick Ruy Lopez or Italian depending on reply
    ("e2e4", "e7e5", "g1f3", "b8c6"): "f1b5",  # 3. Bb5 (Ruy Lopez)

    # If opponent plays something else, we may fall out of book.

(): "e2e4",
    ("e2e4",): "c7c5",

    # 2. Nf3 e6
    ("e2e4", "c7c5"): "g1f3",
    ("e2e4", "c7c5", "g1f3"): "e7e6",

    # 3. d4 cxd4
    ("e2e4", "c7c5", "g1f3", "e7e6"): "d2d4",
    ("e2e4", "c7c5", "g1f3", "e7e6", "d2d4"): "c5d4",

    # 4. Nxd4 a6
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4"): "f3d4",
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4"): "a7a6",

    # 5. Nc3 Qc7
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6"): "b1c3",
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6", "b1c3"): "d8c7",

    # 6. Be3 Nf6
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7"): "f1e3",
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3"): "g8f6",

    # 7. Qd2 Bb4
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6"): "d1d2",
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6",
     "d1d2"): "f8b4",

    # 8. f3 d5
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6",
     "d1d2", "f8b4"): "f2f3",
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6",
     "d1d2", "f8b4", "f2f3"): "d7d5",

    # 9. Bd3 e5
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6",
     "d1d2", "f8b4", "f2f3", "d7d5"): "c1d3",
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6",
     "d1d2", "f8b4", "f2f3", "d7d5",
     "c1d3"): "e6e5",

    # 10. Nde2 d4
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6",
     "d1d2", "f8b4", "f2f3", "d7d5",
     "c1d3", "e6e5"): "d4e2",        # Nde2
    ("e2e4", "c7c5", "g1f3", "e7e6",
     "d2d4", "c5d4", "f3d4", "a7a6",
     "b1c3", "d8c7", "f1e3", "g8f6",
     "d1d2", "f8b4", "f2f3", "d7d5",
     "c1d3", "e6e5", "d4e2"): "d5d4",


    # --- d4 d5 Queen's Gambit ---
    ("d2d4",): "d7d5",  # ... d5
    ("d2d4", "d7d5"): "c2c4",  # 2. c4

    # --- d4 Nf6 2. Nf3 d5 3. Bf4 (London-ish) ---
    ("d2d4", "g8f6"): "g1f3",  # 2. Nf3
    ("d2d4", "g8f6", "g1f3"): "d7d5",  # ... d5
    ("d2d4", "g8f6", "g1f3", "d7d5"): "c1f4",  # 3. Bf4
}


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(board: chess.Board) -> int:
    """
    Simple evaluation function in centipawns.
    Positive = good for side to move.
    We'll use negamax, so this is always "from the side to move".
    """

    # Basic material values
    piece_values = {
        chess.PAWN:   100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK:   500,
        chess.QUEEN:  900,
        chess.KING:   0,   # king material value is irrelevant here
    }

    # Material balance from White's perspective
    score = 0
    for piece_type, value in piece_values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    # Mobility: number of legal moves (encourages activity)
    mobility = board.legal_moves.count()
    if board.turn == chess.WHITE:
        score += 5 * mobility
    else:
        score -= 5 * mobility

    # Convert to "side to move" perspective:
    return score if board.turn == chess.WHITE else -score


# ============================================================
# TRANSPOSITION TABLE HELPERS
# ============================================================

def tt_hash(board: chess.Board) -> int:
    """
    Compute a zobrist hash for the position using python-chess' polyglot helper.
    """
    return polyglot.zobrist_hash(board)


def tt_probe(board: chess.Board, depth: int, alpha: int, beta: int):
    """
    Try to retrieve a useful TT entry.
    Returns (hit, score, flag, stored_move) or (False, None, None, None).
    """
    key = tt_hash(board)
    entry = TT.get(key)
    if entry is None:
        return False, None, None, None

    stored_depth, flag, stored_score, stored_move = entry

    if stored_depth < depth:
        return False, None, None, None

    # Standard TT logic with alpha-beta bounds
    if flag == EXACT:
        return True, stored_score, flag, stored_move
    if flag == LOWERBOUND and stored_score >= beta:
        return True, stored_score, flag, stored_move
    if flag == UPPERBOUND and stored_score <= alpha:
        return True, stored_score, flag, stored_move

    return False, None, None, None


def tt_store(board: chess.Board, depth: int, flag: int, score: int, best_move: Move | None):
    key = tt_hash(board)

    existing = TT.get(key)
    if existing is None or existing[0] <= depth:
        TT[key] = (depth, flag, score, best_move)

    # Very simple size control
    if len(TT) > 200_000:
        TT.clear()


# ============================================================
# SEARCH (NEGAMAX + ALPHA-BETA)
# ============================================================

def order_moves(board: chess.Board, moves, tt_move: Move | None):
    """
    Basic move ordering:
    - TT move first if present
    - Captures next
    - Others after
    """
    def move_key(m: Move):
        if tt_move is not None and m == tt_move:
            return 1000  # highest priority
        if board.is_capture(m):
            return 100   # captures next
        return 0        # others

    return sorted(moves, key=move_key, reverse=True)


def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """
    Negamax with alpha-beta pruning and TT.
    Returns a score from the perspective of the side to move.
    """

    # Check for terminal nodes
    if board.is_game_over():
        # Checkmate vs stalemate/draw
        if board.is_checkmate():
            # Side to move is checkmated -> very bad
            return -INFINITY + 1
        else:
            # Draw or stalemate
            return 0

    if depth == 0:
        return evaluate(board)

    alpha_orig = alpha

    # Probe TT
    tt_hit, tt_score, tt_flag, tt_move = tt_probe(board, depth, alpha, beta)
    if tt_hit:
        return tt_score

    best_score = -INFINITY
    best_move = None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # Should be covered by game_over above, but just in case
        if board.is_check():
            return -INFINITY + 1
        else:
            return 0

    ordered_moves = order_moves(board, legal_moves, tt_move)

    for move in ordered_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score
            if alpha >= beta:
                break  # beta cutoff

    # Store into TT
    if best_score <= alpha_orig:
        flag = UPPERBOUND
    elif best_score >= beta:
        flag = LOWERBOUND
    else:
        flag = EXACT

    tt_store(board, depth, flag, best_score, best_move)

    return best_score


def root_search(board: chess.Board, max_depth: int) -> Move | None:
    """
    Root search with iterative deepening.
    Returns the best move found, or None if no legal moves.
    """

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = random.choice(legal_moves)
    best_score = -INFINITY

    # Simple iterative deepening loop
    for depth in range(1, max_depth + 1):
        current_best_move = None
        current_best_score = -INFINITY

        # Try to get TT move to improve ordering
        tt_hit, _, _, tt_move = tt_probe(board, depth, -INFINITY, INFINITY)
        ordered_moves = order_moves(board, legal_moves, tt_move)

        alpha = -INFINITY
        beta = INFINITY

        for move in ordered_moves:
            board.push(move)
            score = -negamax(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > current_best_score or current_best_move is None:
                current_best_score = score
                current_best_move = move

            if score > alpha:
                alpha = score

        if current_best_move is not None:
            best_move = current_best_move
            best_score = current_best_score

    return best_move


# ============================================================
# OPENING BOOK LOOKUP
# ============================================================

def probe_opening_book(board: chess.Board) -> Move | None:
    """
    Look up a move in the embedded opening book using the move history
    (UCI strings from board.move_stack).
    """
    move_seq = tuple(m.uci() for m in board.move_stack)
    next_move_uci = OPENING_BOOK.get(move_seq)

    if next_move_uci is None:
        return None

    move = Move.from_uci(next_move_uci)
    if move in board.legal_moves:
        return move

    return None


# ============================================================
# ENTRYPOINTS
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called every time the model needs to make a move.
    Must return a python-chess Move that is legal for the current position.
    """

    print("Cooking move...")
    print("Move stack:", [m.uci() for m in ctx.board.move_stack])
    time.sleep(0.01)

    board = ctx.board

    # 1. Try opening book
    book_move = probe_opening_book(board)
    if book_move is not None:
        ctx.logProbabilities({book_move: 1.0})
        return book_move

    # 2. Otherwise, run search
    best_move = root_search(board, MAX_DEPTH)

    if best_move is None or best_move not in board.legal_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (checkmate or stalemate).")
        best_move = random.choice(legal_moves)

    # Log a degenerate distribution: chosen move has probability 1
    ctx.logProbabilities({best_move: 1.0})

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    Clears transposition table and any model state.
    """
    global TT
    TT = {}
    print("New game: transposition table cleared.")