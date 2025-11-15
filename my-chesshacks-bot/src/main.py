from .utils import chess_manager, GameContext
from chess import Move
import chess
import chess.polyglot
import random
import time

# ---------- GLOBALS ----------

# Transposition table:
# key -> (depth, flag, score, best_move)
TT = {}

# Opening book (Polyglot .bin), e.g. "book.bin"
OPENING_BOOK = None

# Search controls
MAX_DEPTH = 3        # start at 3; increase later if it's fast enough
INFINITY = 10_000_000

# TT flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2


# ---------- EVALUATION ----------

def evaluate(board: chess.Board) -> int:
    """
    Simple evaluation function in centipawns.
    Positive = good for side to move.
    We evaluate from the *side to move* perspective via negamax.
    """

    # Basic material values
    piece_values = {
        chess.PAWN:   100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK:   500,
        chess.QUEEN:  900,
        chess.KING:   0,   # material value irrelevant; king safety handled separately
    }

    # Material balance from White's perspective
    score = 0
    for piece_type, value in piece_values.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value

    # Very light positional tweaks (you can expand later)

    # Mobility: number of legal moves (encourages active positions)
    mobility = board.legal_moves.count()
    if board.turn == chess.WHITE:
        score += 5 * mobility
    else:
        score -= 5 * mobility

    # If it's Black to move, flip sign so that positive = good for side to move
    return score if board.turn == chess.WHITE else -score


# ---------- TRANSPOSITION TABLE HELPERS ----------

def tt_hash(board: chess.Board) -> int:
    """
    Compute a zobrist hash for the position.
    Using python-chess' polyglot helper so we don't depend on internal APIs.
    """
    return chess.polyglot.zobrist_hash(board)


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

    # Basic alpha-beta TT logic
    if flag == EXACT:
        return True, stored_score, flag, stored_move
    if flag == LOWERBOUND and stored_score >= beta:
        return True, stored_score, flag, stored_move
    if flag == UPPERBOUND and stored_score <= alpha:
        return True, stored_score, flag, stored_move

    return False, None, None, None


def tt_store(board: chess.Board, depth: int, flag: int, score: int, best_move: Move | None):
    key = tt_hash(board)

    # Simple replacement scheme: always replace if deeper
    existing = TT.get(key)
    if existing is None or existing[0] <= depth:
        TT[key] = (depth, flag, score, best_move)

    # Optional: size limit
    if len(TT) > 200_000:
        # crude aging: just clear if TT gets too big
        TT.clear()


# ---------- SEARCH (NEGAMAX + ALPHA-BETA) ----------

def order_moves(board: chess.Board, moves, tt_move: Move | None):
    """
    Basic move ordering:
    - TT move first if present
    - Captures next
    - Others after
    """
    def move_key(m: Move):
        # TT move highest priority
        if tt_move is not None and m == tt_move:
            return 1000

        # Prioritize captures
        if board.is_capture(m):
            return 100

        # Everything else
        return 0

    return sorted(moves, key=move_key, reverse=True)


def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:
    """
    Negamax with alpha-beta pruning and TT.
    Returns a score from the perspective of the side to move.
    """
    # Check for terminal node or depth limit
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    alpha_original = alpha

    # TT probe
    tt_hit, tt_score, tt_flag, tt_move = tt_probe(board, depth, alpha, beta)
    if tt_hit:
        return tt_score

    best_score = -INFINITY
    best_move = None

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # No legal moves: checkmate or stalemate
        # If checkmated, very bad; otherwise 0.
        if board.is_check():
            # Losing side wants a big negative
            return -INFINITY + 1
        else:
            return 0

    # Move ordering
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

    # Determine TT flag
    if best_score <= alpha_original:
        flag = UPPERBOUND
    elif best_score >= beta:
        flag = LOWERBOUND
    else:
        flag = EXACT

    tt_store(board, depth, flag, best_score, best_move)

    return best_score


def root_search(board: chess.Board, max_depth: int) -> Move:
    """
    Root search with iterative deepening.
    Returns the best move found.
    """
    best_move = None
    best_score = -INFINITY

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    # Iterative deepening
    for depth in range(1, max_depth + 1):
        current_best_move = None
        current_best_score = -INFINITY

        # Optionally use TT move from previous searches
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

        # After this depth, update global best
        if current_best_move is not None:
            best_move = current_best_move
            best_score = current_best_score

    # Fallback (shouldn't happen)
    if best_move is None:
        best_move = random.choice(legal_moves)

    return best_move


# ---------- OPENING BOOK ----------

def probe_opening_book(board: chess.Board) -> Move | None:
    """
    If OPENING_BOOK is loaded and a move exists for this position,
    return a weighted random book move. Otherwise, return None.
    """
    global OPENING_BOOK
    if OPENING_BOOK is None:
        return None

    try:
        entries = list(chess.polyglot.find_all(OPENING_BOOK, board))
    except Exception:
        return None

    if not entries:
        return None

    moves = [entry.move for entry in entries]
    weights = [entry.weight for entry in entries]

    return random.choices(moves, weights=weights, k=1)[0]


# ---------- ENTRYPOINTS ----------

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    """
    Called every time the model needs to make a move.
    Return a python-chess Move object that is legal for the current position.
    """

    print("Cooking move...")
    print("Move stack:", ctx.board.move_stack)
    time.sleep(0.01)  # tiny sleep just so logs don't spam insanely

    board = ctx.board

    # 1. Try opening book first
    book_move = probe_opening_book(board)
    if book_move is not None and book_move in board.legal_moves:
        ctx.logProbabilities({book_move: 1.0})
        return book_move

    # 2. If no book move, run search
    best_move = root_search(board, MAX_DEPTH)

    # Safety fallback
    if best_move is None or best_move not in board.legal_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (checkmate or stalemate).")
        best_move = random.choice(legal_moves)

    # For now, just log a degenerate distribution: chosen move has prob 1
    ctx.logProbabilities({best_move: 1.0})

    return best_move


@chess_manager.reset
def reset_func(ctx: GameContext):
    """
    Called when a new game begins.
    Clears caches, resets model state, reloads opening book, etc.
    """
    global TT, OPENING_BOOK

    # Clear transposition table
    TT = {}

    # Try to load opening book (optional)
    # Put your polyglot book file (e.g. "book.bin") in the same directory,
    # or adjust the path accordingly. If missing, we'll silently ignore.
    try:
        OPENING_BOOK = chess.polyglot.MemoryMappedReader("book.bin")
        print("Opening book loaded.")
    except FileNotFoundError:
        OPENING_BOOK = None
        print("No opening book found; playing without book.")
