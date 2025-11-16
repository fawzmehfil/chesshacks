from src.utils.decorator import chess_manager, GameContext
from chess import Move
import chess
import chess.polyglot as polyglot
import random
import time
import os
from .utils.evaluation import CNNEvaluator
import torch

# ============================================================
# GLOBALS / CONFIG
# ============================================================

TT = {}  # zobrist_key -> (depth, flag, score, best_move)

INFINITY = 10_000_000
MAX_DEPTH = 20  

# TT flags
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

# Basic piece values (for MVV-LVA, etc.)
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0,
}

# ============================================================
# MODEL INITIALIZATION (CNN EVALUATOR)
# ============================================================

model_path = os.path.join(os.path.dirname(__file__), "utils", "chess_cnn_final.pth")
cnn_eval = CNNEvaluator(model_path)

def evaluate(board: chess.Board) -> float:
    score = cnn_eval.evaluate_board(board)
    return score if board.turn == chess.WHITE else -score

# ============================================================
# MVV-LVA TABLE
# ============================================================

MVV_LVA_TABLE = [[0]*7 for _ in range(7)]

MVV_VICTIM_VALUE = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
                    chess.ROOK:5, chess.QUEEN:9, chess.KING:10}
MVV_ATTACKER_VALUE = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
                      chess.ROOK:5, chess.QUEEN:9, chess.KING:10}

for v_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
    for a_type in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        MVV_LVA_TABLE[v_type][a_type] = MVV_VICTIM_VALUE[v_type]*10 - MVV_ATTACKER_VALUE[a_type]

# ============================================================
# TRANSPOSITION TABLE
# ============================================================

def tt_hash(board: chess.Board) -> int:
    return polyglot.zobrist_hash(board)

def tt_probe(board: chess.Board, depth:int, alpha:int, beta:int):
    key = tt_hash(board)
    entry = TT.get(key)
    if entry is None:
        return False, None, None
    stored_depth, flag, stored_score, stored_move = entry
    if stored_depth < depth:
        return False, None, None
    if flag == EXACT: return True, stored_score, stored_move
    if flag == LOWERBOUND and stored_score >= beta: return True, stored_score, stored_move
    if flag == UPPERBOUND and stored_score <= alpha: return True, stored_score, stored_move
    return False, None, None

def tt_store(board: chess.Board, depth:int, flag:int, score:float, best_move:Move|None):
    key = tt_hash(board)
    existing = TT.get(key)
    if existing is None or existing[0]<=depth:
        TT[key]=(depth,flag,score,best_move)
    if len(TT)>200_000:
        TT.clear()

# ============================================================
# MVV-LVA MOVE ORDERING
# ============================================================

def mvv_lva_score(board: chess.Board, move: Move) -> int:
    if board.is_en_passant(move):
        victim_type = chess.PAWN
    else:
        piece = board.piece_at(move.to_square)
        victim_type = piece.piece_type if piece else 0
    attacker_piece = board.piece_at(move.from_square)
    if not attacker_piece: return 0
    attacker_type = move.promotion if move.promotion else attacker_piece.piece_type
    return MVV_LVA_TABLE[victim_type][attacker_type]

def order_moves(board: chess.Board, moves, tt_move: Move|None):
    def key(m:Move):
        score=0
        if tt_move is not None and m==tt_move: score+=100_000
        if board.is_capture(m): score+=1000 + mvv_lva_score(board,m)
        return score
    return sorted(moves,key=key,reverse=True)

# ============================================================
# NULL MOVE PRUNING HELPERS
# ============================================================

NULL_MOVE_REDUCTION = 2

def has_non_pawn_material(board: chess.Board,color:bool)->bool:
    for ptype in (chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN):
        if board.pieces(ptype,color): return True
    return False

def can_do_null_move(board:chess.Board)->bool:
    if board.is_check(): return False
    return has_non_pawn_material(board, board.turn)

# ============================================================
# QUIESCENCE SEARCH WITH TIME CUT
# ============================================================

def quiesce(board: chess.Board, alpha: float, beta: float, start_time: float, max_time: float) -> float:
    if time.time()-start_time>max_time:
        return evaluate(board)
    if board.is_game_over():
        if board.is_checkmate(): return -INFINITY+1
        return 0
    stand_pat = evaluate(board)
    if stand_pat >= beta: return stand_pat
    if stand_pat > alpha: alpha = stand_pat
    captures = [m for m in board.legal_moves if board.is_capture(m)]
    captures.sort(key=lambda m:mvv_lva_score(board,m),reverse=True)
    for move in captures:
        if time.time()-start_time>max_time: break
        board.push(move)
        score = -quiesce(board,-beta,-alpha,start_time,max_time)
        board.pop()
        if score>=beta: return score
        if score>alpha: alpha=score
    return alpha

# ============================================================
# NEGAMAX WITH ALPHA-BETA + NULL MOVE + TIME CUT
# ============================================================

def negamax(board: chess.Board, depth:int, alpha:float, beta:float, allow_null=True,
            start_time: float=None, max_time: float=None)->float:
    if start_time is None: start_time=time.time()
    if max_time is None: max_time=1.0
    if time.time()-start_time>max_time: return evaluate(board)
    if board.is_game_over():
        if board.is_checkmate(): return -INFINITY+1
        return 0
    if depth==0: return quiesce(board,alpha,beta,start_time,max_time)
    alpha_orig=alpha
    if allow_null and depth>=3 and can_do_null_move(board):
        board.push(chess.Move.null())
        null_depth=max(0,depth-1-NULL_MOVE_REDUCTION)
        score=-negamax(board,null_depth,-beta,-beta+1,allow_null=False,start_time=start_time,max_time=max_time)
        board.pop()
        if score>=beta: return score
    tt_hit,tt_score,tt_move=tt_probe(board,depth,alpha,beta)
    if tt_hit: return tt_score
    best_score=-INFINITY
    best_move=None
    legal_moves=list(board.legal_moves)
    if not legal_moves: return -INFINITY+1 if board.is_check() else 0
    ordered_moves=order_moves(board,legal_moves,tt_move)
    for move in ordered_moves:
        if time.time()-start_time>max_time: break
        board.push(move)
        score=-negamax(board,depth-1,-beta,-alpha,allow_null=True,start_time=start_time,max_time=max_time)
        board.pop()
        if score>best_score:
            best_score=score
            best_move=move
        if score>alpha:
            alpha=score
            if alpha>=beta: break
    # Store in TT
    if best_score<=alpha_orig: flag=UPPERBOUND
    elif best_score>=beta: flag=LOWERBOUND
    else: flag=EXACT
    tt_store(board,depth,flag,best_score,best_move)
    return best_score

# ============================================================
# ROOT SEARCH WITH ITERATIVE DEEPENING + TIME CUT
# ============================================================

def root_search(board: chess.Board, ctx: GameContext, max_depth:int=MAX_DEPTH) -> Move|None:
    legal_moves=list(board.legal_moves)
    if not legal_moves: return None
    best_move=random.choice(legal_moves)
    start_time=time.time()
    max_time=min(1.0, ctx.timeLeft/1000*0.9)
    for depth in range(1,max_depth+1):
        current_best_move=None
        current_best_score=-INFINITY
        tt_hit,_,tt_move=tt_probe(board,depth,-INFINITY,INFINITY)
        ordered_moves=order_moves(board,legal_moves,tt_move)
        alpha=-INFINITY
        beta=INFINITY
        for move in ordered_moves:
            if time.time()-start_time>max_time: break
            board.push(move)
            score=-negamax(board,depth,-beta,-alpha,allow_null=True,start_time=start_time,max_time=max_time)
            board.pop()
            if score>current_best_score:
                current_best_score=score
                current_best_move=move
            if score>alpha: alpha=score
        if current_best_move is not None:
            best_move=current_best_move
        if time.time()-start_time>max_time: break
    return best_move

# ============================================================
# ENTRYPOINTS
# ============================================================

@chess_manager.entrypoint
def test_func(ctx: GameContext):
    print("Cooking move...")
    print("Move stack:", [m.uci() for m in ctx.board.move_stack])
    board=ctx.board
    best_move=root_search(board,ctx,MAX_DEPTH)
    if best_move is None or best_move not in board.legal_moves:
        legal_moves=list(board.legal_moves)
        if not legal_moves:
            ctx.logProbabilities({})
            raise ValueError("No legal moves available (checkmate or stalemate).")
        best_move=random.choice(legal_moves)
    ctx.logProbabilities({best_move:1.0})
    return best_move

@chess_manager.reset
def reset_func(ctx: GameContext):
    global TT
    TT={}
    print("New game: transposition table cleared.")
