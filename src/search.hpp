#pragma once

#include "board.hpp"
#include "evaluation.hpp"
#include "see.hpp"

// ======================== Search (ID + TT + QS + Ordering) ========================
struct SearchStats {
    u64 nodes=0;
    u64 qnodes=0;
    int depthReached=0;
    int bestScore=0;
    int timeMs=0;
    int softTimeLimitMs=0;
    int hardTimeLimitMs=0;
    int bestMoveChanges=0;
    int aspirationResearches=0;
    int configuredThreads=1;
    int workersUsed=1;
    int hardwareThreads=1;
};

enum class MoveStage : u8;

// Reused heap storage, indexed by active call nesting (not chess ply: IID and
// null verification can re-enter at the same ply). Pointers stay stable on growth.
struct SearchScratch {
    std::array<MoveList, 3> lists;
    std::array<int, 320> scores{};
    std::array<MoveStage, 320> stages{};
};

struct SearchContext {
    static constexpr int MaxSearchPly = 127;
    static constexpr size_t NnueStackSize = 129;
    TranspositionTable tt;
    TranspositionTable* sharedTT=nullptr;
    SearchStats stats;
    std::chrono::steady_clock::time_point start;
    int softTimeLimitMs=1000;
    int hardTimeLimitMs=1000;
    bool stop=false;
    u32 timeCheckCounter=0;
    const std::atomic<bool>* abortFlag=nullptr;
    const PositionEvaluator* evaluator=nullptr;
    bool incrementalNnue=true;
    std::array<NnueAccumulator, NnueStackSize> nnueStack{};

    Move killer[128][2]{};
    Move countermove[2][64][64]{};
    Move plyMove[128]{};
    int history[2][64][64]{};
    int continuationHistory[2][64][64]{};
    int captureHistory[2][7][64]{};
    int staticEvalByPly[128]{};
    std::vector<u64> gameHistory; // position hashes from actual game (includes current root)
    std::vector<u64> repetition;
    std::vector<std::unique_ptr<SearchScratch>> scratch;
    size_t activeScratch=0;
};

class SearchScratchScope {
public:
    explicit SearchScratchScope(SearchContext& context) : context_(context){
        const size_t index = context_.activeScratch;
        if(index == context_.scratch.size())
            context_.scratch.push_back(std::make_unique<SearchScratch>());
        frame_ = context_.scratch[index].get();
        for(auto& list : frame_->lists) list.clear();
        context_.activeScratch++;
    }
    ~SearchScratchScope(){ context_.activeScratch--; }
    SearchScratchScope(const SearchScratchScope&) = delete;
    SearchScratchScope& operator=(const SearchScratchScope&) = delete;
    SearchScratch& frame(){ return *frame_; }
private:
    SearchContext& context_;
    SearchScratch* frame_;
};

inline TranspositionTable& searchTT(SearchContext& context){
    return context.sharedTT ? *context.sharedTT : context.tt;
}

class RootWorkerPool {
public:
    explicit RootWorkerPool(int workerCount)
        : pendingWorkers(0), stopping(false), generation(0){
        threads.reserve(static_cast<size_t>(workerCount));
        for(int worker = 0; worker < workerCount; worker++){
            threads.emplace_back([this, worker](){ workerLoop(worker); });
        }
    }

    RootWorkerPool(const RootWorkerPool&) = delete;
    RootWorkerPool& operator=(const RootWorkerPool&) = delete;

    ~RootWorkerPool(){
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
            generation++;
        }
        workReady.notify_all();
        for(std::thread& thread : threads){
            if(thread.joinable()) thread.join();
        }
    }

    void run(const std::function<void(int)>& nextTask){
        {
            std::lock_guard<std::mutex> lock(mutex);
            task = nextTask;
            pendingWorkers = static_cast<int>(threads.size());
            generation++;
        }
        workReady.notify_all();

        std::unique_lock<std::mutex> lock(mutex);
        workFinished.wait(lock, [&](){ return pendingWorkers == 0; });
        task = {};
    }

private:
    void workerLoop(int worker){
        size_t observedGeneration = 0;
        while(true){
            std::function<void(int)> currentTask;
            {
                std::unique_lock<std::mutex> lock(mutex);
                workReady.wait(lock, [&](){
                    return stopping || generation != observedGeneration;
                });
                if(stopping) return;
                observedGeneration = generation;
                currentTask = task;
            }

            currentTask(worker);

            {
                std::lock_guard<std::mutex> lock(mutex);
                pendingWorkers--;
                if(pendingWorkers == 0) workFinished.notify_one();
            }
        }
    }

    std::vector<std::thread> threads;
    std::mutex mutex;
    std::condition_variable workReady;
    std::condition_variable workFinished;
    std::function<void(int)> task;
    int pendingWorkers;
    bool stopping;
    size_t generation;
};

inline int evaluatePosition(const Board& board, const SearchContext& context, int ply){
    if(!context.evaluator) return evaluateClassical(board);
    if(!context.incrementalNnue) return context.evaluator->evaluate(board);
    if(ply >= 0 && static_cast<size_t>(ply) < SearchContext::NnueStackSize){
        return context.evaluator->evaluate(board, context.nnueStack[static_cast<size_t>(ply)]);
    }
    return context.evaluator->evaluate(board);
}

inline void refreshNnueRoot(const Board& board, SearchContext& context){
    if(context.evaluator && context.incrementalNnue){
        context.evaluator->refreshAccumulator(board, context.nnueStack[0]);
    }
}

inline void advanceNnue(const Board& board, const Undo& undo,
                        SearchContext& context, int parentPly){
    if(!context.evaluator || !context.incrementalNnue || parentPly < 0) return;
    const size_t parent = static_cast<size_t>(parentPly);
    const size_t child = parent + 1;
    if(child >= SearchContext::NnueStackSize) return;
    context.evaluator->updateAccumulator(
        board, undo, context.nnueStack[parent], context.nnueStack[child]);
}

inline void advanceNnueNull(SearchContext& context, int parentPly){
    if(!context.evaluator || !context.incrementalNnue || parentPly < 0) return;
    const size_t parent = static_cast<size_t>(parentPly);
    const size_t child = parent + 1;
    if(child >= SearchContext::NnueStackSize) return;
    context.evaluator->copyAccumulator(context.nnueStack[parent], context.nnueStack[child]);
}

inline bool sameMove(const Move& a, const Move& b){
    return a.from==b.from && a.to==b.to && a.promo==b.promo && a.isCastle==b.isCastle && a.isEnPassant==b.isEnPassant;
}

inline Move invalidMove(){
    Move m{};
    m.from = 64;
    m.to = 64;
    return m;
}

inline int mvvLvaScore(const Board& bd, const Move& m){
    Piece a = bd.at(m.from);
    int attacker = pieceValue(a.t);
    int victim = 0;
    if(m.isEnPassant){
        victim = pieceValue(PieceType::Pawn);
    } else if(m.isCapture){
        Piece v = bd.at(m.to);
        victim = pieceValue(v.t);
    }
    return victim*10 - attacker;
}

inline int scoreCaptureMove(const Board& bd, SearchContext& ctx, const Move& m, int see){
    const int side = (bd.stm==Color::White)?0:1;
    const Piece attacker = bd.at(m.from);
    const int attackerType = std::clamp(int(attacker.t), 0, 6);
    const int band = see >= 0 ? 100000 : 70000;
    return band + see * 8 + mvvLvaScore(bd, m) +
           ctx.captureHistory[side][attackerType][m.to];
}

inline int scoreMove(const Board& bd, SearchContext& ctx, const Move& m, const Move& ttMove, int ply, const Move& prevMove){
    if(ttMove.from < 64 && ttMove.from==m.from && ttMove.to==m.to && ttMove.promo==m.promo) return 1000000;

    const int side = (bd.stm==Color::White)?0:1;
    const Piece attacker = bd.at(m.from);
    const int attackerType = std::clamp(int(attacker.t), 0, 6);

    if(m.promo != PieceType::None){
        int s = 140000 + pieceValue(m.promo) * 4;
        if(m.isCapture || m.isEnPassant){
            s += mvvLvaScore(bd, m);
            s += ctx.captureHistory[side][attackerType][m.to];
        }
        return s;
    }

    if(m.isCapture || m.isEnPassant){
        const int see = staticExchangeEvaluation(bd, m);
        return scoreCaptureMove(bd, ctx, m, see);
    }

    if(ply<128){
        if(sameMove(m, ctx.killer[ply][0])) return 90000;
        if(sameMove(m, ctx.killer[ply][1])) return 80000;
    }

    if(prevMove.from < 64 && prevMove.to < 64){
        const Move& cm = ctx.countermove[side][prevMove.from][prevMove.to];
        if(sameMove(m, cm)) return 85000;
    }
    int quietScore = ctx.history[side][m.from][m.to];
    if(prevMove.to < 64){
        quietScore += ctx.continuationHistory[side][prevMove.to][m.to];
    }
    return quietScore;
}

enum class MoveStage : u8 {
    Transposition,
    GoodTactical,
    SpecialQuiet,
    Quiet,
    BadTactical,
    Done
};

class StagedMovePicker {
public:
    StagedMovePicker(const Board& board, SearchContext& context, MoveList& moves,
                     const Move& ttMove, int ply, const Move& previousMove)
        : scratch_(context), moves_(moves), scores_(scratch_.frame().scores), stages_(scratch_.frame().stages){
        const int side = board.stm == Color::White ? 0 : 1;
        const Move counter = previousMove.from < 64 && previousMove.to < 64
            ? context.countermove[side][previousMove.from][previousMove.to]
            : invalidMove();

        for(size_t index = 0; index < moves_.size(); index++){
            const Move& move = moves_[index];
            if(ttMove.from < 64 && sameMove(move, ttMove)){
                stages_[index] = MoveStage::Transposition;
                scores_[index] = 1000000;
            } else if(move.promo != PieceType::None){
                stages_[index] = MoveStage::GoodTactical;
                scores_[index] = scoreMove(board, context, move, invalidMove(), ply, previousMove);
            } else if(move.isCapture || move.isEnPassant){
                const int see = staticExchangeEvaluation(board, move);
                stages_[index] = see >= 0 ? MoveStage::GoodTactical : MoveStage::BadTactical;
                scores_[index] = scoreCaptureMove(board, context, move, see);
            } else {
                const bool killer = ply < 128 &&
                    (sameMove(move, context.killer[ply][0]) ||
                     sameMove(move, context.killer[ply][1]));
                stages_[index] = killer || sameMove(move, counter)
                    ? MoveStage::SpecialQuiet
                    : MoveStage::Quiet;
                scores_[index] = scoreMove(board, context, move, invalidMove(), ply, previousMove);
            }
        }
    }

    bool next(Move& move){
        while(stage_ != MoveStage::Done){
            size_t bestIndex = moves_.size();
            int bestScore = -1000000000;
            for(size_t index = nextIndex_; index < moves_.size(); index++){
                if(stages_[index] != stage_ || scores_[index] <= bestScore) continue;
                bestIndex = index;
                bestScore = scores_[index];
            }
            if(bestIndex == moves_.size()){
                stage_ = static_cast<MoveStage>(static_cast<u8>(stage_) + 1);
                continue;
            }

            std::swap(moves_[nextIndex_], moves_[bestIndex]);
            std::swap(scores_[nextIndex_], scores_[bestIndex]);
            std::swap(stages_[nextIndex_], stages_[bestIndex]);
            move = moves_[nextIndex_++];
            return true;
        }
        return false;
    }

private:
    SearchScratchScope scratch_;
    MoveList& moves_;
    std::array<int, 320>& scores_;
    std::array<MoveStage, 320>& stages_;
    size_t nextIndex_=0;
    MoveStage stage_=MoveStage::Transposition;
};

inline void updateHistoryValue(int& entry, int bonus){
    constexpr int HistoryLimit = 90000;
    bonus = std::clamp(bonus, -HistoryLimit, HistoryLimit);
    entry += bonus - (entry * std::abs(bonus)) / HistoryLimit;
    entry = std::clamp(entry, -HistoryLimit, HistoryLimit);
}

template<typename MoveContainer, typename Scorer>
inline void sortMovesByScore(MoveContainer& moves, Scorer scorer){
    struct ScoredMove { int score; Move move; };
    thread_local std::vector<ScoredMove> scored;
    scored.clear();
    if(scored.capacity() < 256) scored.reserve(256);
    for(const Move& move : moves) scored.push_back(ScoredMove{scorer(move), move});
    std::sort(scored.begin(), scored.end(), [](const ScoredMove& lhs, const ScoredMove& rhs){
        return lhs.score > rhs.score;
    });
    for(size_t index = 0; index < scored.size(); index++) moves[index] = scored[index].move;
}

inline bool timeUp(SearchContext& ctx){
    if(ctx.stop) return true;
    if(ctx.abortFlag && ctx.abortFlag->load(std::memory_order_relaxed)){
        ctx.stop = true;
        return true;
    }
    // Check immediately on entry, then more frequently for emergency budgets.
    // The first check must include time spent waiting for the UCI worker.
    const u32 mask = ctx.hardTimeLimitMs <= 10 ? 0U
                   : ctx.hardTimeLimitMs <= 100 ? 31U : 255U;
    if((ctx.timeCheckCounter++ & mask) != 0) return false;
    auto now = std::chrono::steady_clock::now();
    int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.start).count();
    if(ms >= ctx.hardTimeLimitMs){
        ctx.stop=true;
        return true;
    }
    return false;
}

inline int elapsedTimeMs(const SearchContext& ctx){
    auto now = std::chrono::steady_clock::now();
    return (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.start).count();
}

inline bool softTimeUp(const SearchContext& ctx){
    return elapsedTimeMs(ctx) >= ctx.softTimeLimitMs;
}

inline const int INF = 100000000;
inline const int MATE = 1000000;

inline int scoreToTT(int score, int ply){
    if(score >= MATE - 10000) return score + ply;
    if(score <= -MATE + 10000) return score - ply;
    return score;
}

inline int scoreFromTT(int score, int ply){
    if(score >= MATE - 10000) return score - ply;
    if(score <= -MATE + 10000) return score + ply;
    return score;
}

inline bool hasNonPawnMaterial(const Board& bd, Color side){
    for(const auto& p : bd.b){
        if(isNone(p) || p.c != side) continue;
        if(p.t != PieceType::King && p.t != PieceType::Pawn) return true;
    }
    return false;
}

inline int nonKingPieceCount(const Board& bd, Color side){
    int count = 0;
    for(const auto& p : bd.b){
        if(isNone(p) || p.c != side || p.t == PieceType::King) continue;
        count++;
    }
    return count;
}

inline bool isThreefoldRepetition(const Board& bd, const SearchContext& ctx){
    if(ctx.repetition.empty()) return false;

    const size_t n = ctx.repetition.size();
    const size_t maxBack = std::min<size_t>(size_t(std::max(0, bd.halfmoveClock)), n - 1);

    int seen = 0;
    for(size_t back = 0; back <= maxBack; back++){
        const size_t idx = n - 1 - back;
        if(ctx.repetition[idx] == bd.hash){
            seen++;
            if(seen >= 3) return true;
        }
    }
    return false;
}

inline int quiescence(Board& bd, SearchContext& ctx, int alpha, int beta, int ply){
    if(timeUp(ctx)) return 0;
    SearchScratchScope scratch(ctx);
    ctx.stats.qnodes++;

    const bool inCheck = bd.inCheck(bd.stm);
    const bool ruleDraw = bd.halfmoveClock >= 100 || isThreefoldRepetition(bd, ctx);
    if(ruleDraw){
        // A checkmated position is terminal before a draw can be claimed.
        if(inCheck){
            auto& evasions = scratch.frame().lists[0];
            bd.genLegalMoves(evasions);
            if(evasions.empty()) return -MATE + ply;
        }
        return 0;
    }
    if(bd.insufficientMaterial()) return 0;

    // Check extensions and check evasions can otherwise recurse indefinitely
    // through pathological checking cycles. Keep every search inside the
    // fixed per-ply state arrays and the process stack.
    if(ply >= SearchContext::MaxSearchPly){
        if(inCheck){
            auto& evasions = scratch.frame().lists[0];
            bd.genLegalMoves(evasions);
            if(evasions.empty()) return -MATE + ply;
            return 0;
        }
        return evaluatePosition(bd, ctx, ply);
    }

    alpha = std::max(alpha, -MATE + ply);
    beta = std::min(beta, MATE - ply - 1);
    if(alpha >= beta) return alpha;

    if(inCheck){
        auto& evasions = scratch.frame().lists[0];
        bd.genLegalMoves(evasions);
        if(evasions.empty()) return -MATE + ply;

        int best = -INF;
        for(const Move& m : evasions){
            Undo u{};
            if(!bd.makeMove(m, u)) continue;
            advanceNnue(bd, u, ctx, ply);
            ctx.repetition.push_back(bd.hash);
            int score = -quiescence(bd, ctx, -beta, -alpha, ply + 1);
            ctx.repetition.pop_back();
            bd.undoMove(u);

            if(score > best) best = score;
            if(score > alpha) alpha = score;
            if(alpha >= beta) return score;
        }
        return best;
    }

    int stand = evaluatePosition(bd, ctx, ply);
    if(stand >= beta) return stand;
    if(stand > alpha) alpha = stand;

    auto& pseudo = scratch.frame().lists[0];
    bd.genPseudoMoves(pseudo);

    auto& moves = scratch.frame().lists[1];
    moves.reserve(pseudo.size());
    for(const Move& m : pseudo){
        if(!(m.isCapture || m.isEnPassant || m.promo != PieceType::None)) continue;

        if(!m.isEnPassant && m.promo == PieceType::None){
            int victim = 0;
            if(m.isCapture){
                const Piece v = bd.at(m.to);
                victim = pieceValue(v.t);
            }
            const int deltaMargin = 120;
            if(stand + victim + deltaMargin < alpha){
                continue;
            }

            // Deeply losing exchanges rarely improve a quiet stand-pat score.
            // Keep a small negative allowance for tactical uncertainty while
            // removing obviously bad captures before ordering and recursion.
            if(staticExchangeEvaluation(bd, m) < -80){
                continue;
            }
        }
        moves.push_back(m);
    }

    sortMovesByScore(moves, [&](const Move& move){
        int score = staticExchangeEvaluation(bd, move) * 8 + mvvLvaScore(bd, move);
        if(move.promo != PieceType::None) score += 2000 + pieceValue(move.promo);
        return score;
    });

    for(const Move& m : moves){
        Undo u{};
        if(!bd.makeMove(m, u)) continue;
        advanceNnue(bd, u, ctx, ply);
        ctx.repetition.push_back(bd.hash);
        int score = -quiescence(bd, ctx, -beta, -alpha, ply + 1);
        ctx.repetition.pop_back();
        bd.undoMove(u);

        if(score >= beta) return score;
        if(score > alpha) alpha = score;
    }

    return alpha;
}

inline int negamax(Board& bd, SearchContext& ctx, int depth, int alpha, int beta, int ply, const Move& prevMove, bool allowNullMove){
    if(timeUp(ctx)) return 0;
    SearchScratchScope scratch(ctx);
    ctx.stats.nodes++;

    if(ply >= SearchContext::MaxSearchPly){
        const bool inCheck = bd.inCheck(bd.stm);
        if(inCheck){
            auto& evasions = scratch.frame().lists[0];
            bd.genLegalMoves(evasions);
            if(evasions.empty()) return -MATE + ply;
            return 0;
        }
        if(bd.halfmoveClock >= 100 || isThreefoldRepetition(bd, ctx) ||
           bd.insufficientMaterial()){
            return 0;
        }
        return evaluatePosition(bd, ctx, ply);
    }

    // Mate-distance pruning keeps mate scores consistent and trims impossible windows.
    alpha = std::max(alpha, -MATE + ply);
    beta = std::min(beta, MATE - ply - 1);
    if(alpha >= beta) return alpha;
    const bool pvNode = (beta - alpha) > 1;

    const bool ruleDraw = bd.halfmoveClock >= 100 || isThreefoldRepetition(bd, ctx);
    if(ruleDraw){
        // Checkmate ends the game before a draw claim can be made.
        if(bd.inCheck(bd.stm)){
            auto& evasions = scratch.frame().lists[0];
            bd.genLegalMoves(evasions);
            if(evasions.empty()) return -MATE + ply;
        }
        return 0;
    }
    if(bd.insufficientMaterial()) return 0;

    bool inCheck = bd.inCheck(bd.stm);
    int staticEval = 0;
    if(!inCheck){
        staticEval = evaluatePosition(bd, ctx, ply);
        if(ply < 128) ctx.staticEvalByPly[ply] = staticEval;
    } else if(ply < 128){
        ctx.staticEvalByPly[ply] = -INF;
    }

    bool improving = false;
    if(!inCheck && ply >= 2 && ply < 128){
        improving = staticEval > ctx.staticEvalByPly[ply - 2];
    }

    if(!inCheck && depth <= 3){
        const int rfpMargin = 95 * depth;
        if(staticEval - rfpMargin >= beta){
            return staticEval;
        }
    }

    Move ttMove = invalidMove();
    TranspositionTable& tt = searchTT(ctx);
    if(const auto entry = tt.probe(bd.hash)){
        const TTEntry& e = *entry;
        if(e.key==bd.hash){
            ttMove = e.best;
            if(e.depth >= depth){
                int s = scoreFromTT(e.score, ply);
                if(e.flag==TTFlag::Exact) return s;
                if(e.flag==TTFlag::Lower) alpha = std::max(alpha, s);
                else if(e.flag==TTFlag::Upper) beta = std::min(beta, s);
                if(alpha >= beta) return s;
            }
        }
    }

    if(!inCheck && depth >= 6 && ttMove.from >= 64){
        const int iidDepth = std::max(1, depth - 2);
        (void)negamax(bd, ctx, iidDepth, alpha, beta, ply, prevMove, false);
        if(ctx.stop) return 0;
        if(const auto entry = tt.probe(bd.hash)){
            if(entry->key == bd.hash){
                ttMove = entry->best;
            }
        }
    }

    if(depth <= 0){
        return quiescence(bd, ctx, alpha, beta, ply);
    }

    if(!inCheck && !pvNode && depth <= 2){
        const int razorMargin = 180 + 120 * depth;
        if(staticEval + razorMargin <= alpha){
            return quiescence(bd, ctx, alpha, beta, ply);
        }
    }

    // Null-move pruning: aggressive cut when position is quiet enough and side has material.
    if(allowNullMove && !pvNode && depth >= 3 && !inCheck &&
       hasNonPawnMaterial(bd, bd.stm) && nonKingPieceCount(bd, bd.stm) >= 2){
        NullUndo nullUndo{};
        bd.makeNullMove(nullUndo);
        advanceNnueNull(ctx, ply);
        ctx.repetition.push_back(bd.hash);
        int reduction = 2 + depth / 4;
        if(depth >= 7) reduction++;
        if(!improving) reduction++;
        if(staticEval >= beta + 120) reduction++;
        reduction = std::clamp(reduction, 2, std::max(2, depth - 2));
        int score = -negamax(bd, ctx, depth - 1 - reduction, -beta, -beta + 1, ply + 1, invalidMove(), false);
        ctx.repetition.pop_back();
        bd.undoNullMove(nullUndo);
        if(ctx.stop) return 0;
        if(score >= beta){
            if(depth >= 6){
                const int verifyDepth = std::max(0, depth - 1 - reduction);
                int verify = negamax(bd, ctx, verifyDepth, beta - 1, beta, ply, prevMove, false);
                if(ctx.stop) return 0;
                if(verify >= beta) return beta;
            } else {
                return beta;
            }
        }
    }

    auto& moves = scratch.frame().lists[0];
    bd.genLegalMoves(moves);

    if(moves.empty()){
        if(bd.inCheck(bd.stm)) return -MATE + ply;
        return 0;
    }

    if(inCheck && moves.size() == 1 && depth < 10){
        depth++;
    }

    StagedMovePicker movePicker(bd, ctx, moves, ttMove, ply, prevMove);

    int best = -INF;
    Move bestM{};

    int originalAlpha = alpha;
    const int side = (bd.stm==Color::White)?0:1;
    auto& quietTried = scratch.frame().lists[1];
    auto& tacticalTried = scratch.frame().lists[2];
    quietTried.reserve(moves.size());
    tacticalTried.reserve(moves.size());

    Move m{};
    size_t i=0;
    while(movePicker.next(m)){
        const size_t moveIndex = i++;
        bool isQuiet = !(m.isCapture || m.isEnPassant) && (m.promo==PieceType::None);

        if(!inCheck && !pvNode && isQuiet){
            if(depth <= 3){
                const size_t futilitySkipAfter = size_t(4 + depth * 3);
                const int futilityMargin = 90 + 120 * depth + (improving ? 20 : 0);
                if(moveIndex >= futilitySkipAfter && staticEval + futilityMargin <= alpha){
                    continue;
                }
            }

            if(depth <= 4){
                const size_t lmpThreshold = size_t(3 + depth * depth + (improving ? 2 : 0));
                if(moveIndex >= lmpThreshold){
                    continue;
                }
            }
        }

        if(isQuiet) quietTried.push_back(m);
        else tacticalTried.push_back(m);

        Undo u{};
        if(!bd.makeMove(m,u)) continue;
        advanceNnue(bd, u, ctx, ply);

        if(ply < 128) ctx.plyMove[ply] = m;
        ctx.repetition.push_back(bd.hash);

        int newDepth = depth - 1;
        bool givesCheck = bd.inCheck(bd.stm);
        if(givesCheck){
            newDepth++;
        }
        int score = 0;

        int reduction = 0;
        if(newDepth >= 3 && moveIndex >= 3 && isQuiet && !givesCheck){
            reduction = 1;
            if(moveIndex >= 4) reduction++;
            if(moveIndex >= 8) reduction++;
            if(newDepth >= 5) reduction++;
            if(newDepth >= 8 && moveIndex >= 12) reduction++;
            if(!improving) reduction++;
            if(pvNode) reduction--;
            if(ply < 128 && sameMove(m, ctx.killer[ply][0])) reduction--;
            if(ctx.history[side][m.from][m.to] > 18000) reduction -= 2;
            else if(ctx.history[side][m.from][m.to] > 9000) reduction--;
            reduction = std::clamp(reduction, 0, std::max(0, newDepth - 1));
        }

        if(moveIndex == 0){
            score = -negamax(bd, ctx, newDepth, -beta, -alpha, ply + 1, m, true);
        } else {
            // PVS + LMR: search late quiet moves reduced on a null-window first.
            const int scoutDepth = std::max(0, newDepth - reduction);
            score = -negamax(bd, ctx, scoutDepth, -alpha - 1, -alpha, ply + 1, m, true);

            if(!ctx.stop && reduction > 0 && score > alpha){
                score = -negamax(bd, ctx, newDepth, -alpha - 1, -alpha, ply + 1, m, true);
            }

            if(!ctx.stop && score > alpha && score < beta){
                score = -negamax(bd, ctx, newDepth, -beta, -alpha, ply + 1, m, true);
            }
        }

        ctx.repetition.pop_back();
        bd.undoMove(u);

        if(ctx.stop) return 0;

        if(score > best){
            best = score;
            bestM = m;
        }

        alpha = std::max(alpha, score);
        if(alpha >= beta){
            if(isQuiet && ply<128){
                if(!sameMove(ctx.killer[ply][0], m)){
                    ctx.killer[ply][1] = ctx.killer[ply][0];
                    ctx.killer[ply][0] = m;
                }
                const int bonus = depth * depth * 16;
                updateHistoryValue(ctx.history[side][m.from][m.to], bonus);
                if(prevMove.to < 64){
                    updateHistoryValue(ctx.continuationHistory[side][prevMove.to][m.to], bonus);
                }
                for(const Move& qm : quietTried){
                    if(sameMove(qm, m)) continue;
                    updateHistoryValue(ctx.history[side][qm.from][qm.to], -(bonus / 2));
                    if(prevMove.to < 64){
                        updateHistoryValue(ctx.continuationHistory[side][prevMove.to][qm.to], -(bonus / 2));
                    }
                }
                if(prevMove.from < 64 && prevMove.to < 64){
                    ctx.countermove[side][prevMove.from][prevMove.to] = m;
                }
            } else if(ply < 128){
                const Piece attacker = bd.at(m.from);
                const int attackerType = std::clamp(int(attacker.t), 0, 6);
                const int bonus = depth * depth * 20;
                int& slot = ctx.captureHistory[side][attackerType][m.to];
                slot = std::min(90000, slot + bonus);

                for(const Move& tm : tacticalTried){
                    if(sameMove(tm, m)) continue;
                    const Piece triedPiece = bd.at(tm.from);
                    const int triedType = std::clamp(int(triedPiece.t), 0, 6);
                    int& penaltySlot = ctx.captureHistory[side][triedType][tm.to];
                    penaltySlot = std::max(-90000, penaltySlot - (bonus / 3));
                }
            }
            break;
        }
    }

    TTFlag flag = TTFlag::Exact;
    if(best <= originalAlpha) flag = TTFlag::Upper;
    else if(best >= beta) flag = TTFlag::Lower;
    tt.store(bd.hash, depth, scoreToTT(best, ply), flag, bestM);

    return best;
}

inline Move searchBestMoveSingle(Board& bd, SearchContext& ctx, int maxDepth, int softTimeLimitMs, int hardTimeLimitMs,
                                 const std::vector<Move>* rootRestriction = nullptr,
                                 std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now()){
    ctx.stats = {};
    ctx.stats.softTimeLimitMs = softTimeLimitMs;
    ctx.stats.hardTimeLimitMs = hardTimeLimitMs;
    ctx.stats.configuredThreads = 1;
    ctx.stats.workersUsed = 1;
    ctx.stats.hardwareThreads = std::max(1, int(std::thread::hardware_concurrency()));
    ctx.start = started;
    ctx.softTimeLimitMs = softTimeLimitMs;
    ctx.hardTimeLimitMs = std::max(softTimeLimitMs, hardTimeLimitMs);
    ctx.stop = false;
    ctx.timeCheckCounter = 0;
    searchTT(ctx).newSearch();
    const Move noMove = invalidMove();

    for(int s=0; s<2; s++){
        for(int from=0; from<64; from++){
            for(int to=0; to<64; to++){
                ctx.history[s][from][to] = (ctx.history[s][from][to] * 7) / 8;
                ctx.continuationHistory[s][from][to] =
                    (ctx.continuationHistory[s][from][to] * 7) / 8;
            }
        }
        for(int pt=0; pt<7; pt++){
            for(int to=0; to<64; to++){
                ctx.captureHistory[s][pt][to] = (ctx.captureHistory[s][pt][to] * 7) / 8;
            }
        }
    }
    for(int ply = 0; ply < 128; ply++){
        ctx.plyMove[ply] = noMove;
        ctx.staticEvalByPly[ply] = -INF;
    }

    ctx.repetition = ctx.gameHistory;
    if(ctx.repetition.empty() || ctx.repetition.back() != bd.hash){
        ctx.repetition.push_back(bd.hash);
    }
    refreshNnueRoot(bd, ctx);

    std::vector<Move> rootMoves;
    bd.genLegalMoves(rootMoves);
    if(rootRestriction){
        rootMoves.erase(std::remove_if(rootMoves.begin(), rootMoves.end(), [&](const Move& move){
            return std::none_of(rootRestriction->begin(), rootRestriction->end(), [&](const Move& allowed){
                return sameMove(move, allowed);
            });
        }), rootMoves.end());
    }
    if(rootMoves.empty()) return Move{};

    Move bestMove = rootMoves[0];
    int bestScore = 0;
    int bestMoveChanges = 0;
    int aspirationResearches = 0;
    int dynamicSoftLimitMs = ctx.softTimeLimitMs;

    for(int d=1; d<=maxDepth; d++){
        if(timeUp(ctx)) break;
        if(d > 1 && d >= 4 && softTimeUp(ctx)) break;

        int window = INF;
        if(d >= 3 && std::abs(bestScore) < MATE / 2){
            window = 40;
        }

        int acceptedScore = -INF;
        Move acceptedMove = bestMove;

        while(!ctx.stop){
            if(timeUp(ctx)) break;

            int alpha = -INF;
            int beta = INF;
            const bool aspiration = (window < INF);
            if(aspiration){
                alpha = bestScore - window;
                beta = bestScore + window;
            }

            Move ttMove = noMove;
            if(const auto entry = searchTT(ctx).probe(bd.hash)){
                if(entry->key==bd.hash) ttMove = entry->best;
            }

            sortMovesByScore(rootMoves, [&](const Move& move){
                int score = scoreMove(bd, ctx, move, ttMove, 0, noMove);
                if(sameMove(move, bestMove)) score += 200000;
                return score;
            });

            int localBest = -INF;
            Move localMove = rootMoves[0];
            int alphaRun = alpha;

            for(size_t i=0; i<rootMoves.size(); i++){
                const Move& m = rootMoves[i];
                if(timeUp(ctx)) break;
                Undo u{};
                if(!bd.makeMove(m,u)) continue;
                advanceNnue(bd, u, ctx, 0);

                ctx.repetition.push_back(bd.hash);
                int score = 0;
                if(i == 0){
                    score = -negamax(bd, ctx, d - 1, -beta, -alphaRun, 1, m, true);
                } else {
                    score = -negamax(bd, ctx, d - 1, -alphaRun - 1, -alphaRun, 1, m, true);
                    if(!ctx.stop && score > alphaRun && score < beta){
                        score = -negamax(bd, ctx, d - 1, -beta, -alphaRun, 1, m, true);
                    }
                }
                ctx.repetition.pop_back();
                bd.undoMove(u);

                if(ctx.stop) break;

                if(score > localBest){
                    localBest = score;
                    localMove = m;
                }
                alphaRun = std::max(alphaRun, score);
                if(alphaRun >= beta){
                    break;
                }
            }

            if(ctx.stop) break;
            if(localBest == -INF) break;

            const bool failLow = aspiration && (localBest <= alpha);
            const bool failHigh = aspiration && (localBest >= beta);
            if(failLow || failHigh){
                aspirationResearches++;
                if(window >= INF / 4){
                    window = INF;
                } else {
                    window *= 2;
                }
                continue;
            }

            acceptedScore = localBest;
            acceptedMove = localMove;
            break;
        }

        if(!ctx.stop && acceptedScore != -INF){
            const Move previousBest = bestMove;
            const int previousScore = bestScore;
            bestScore = acceptedScore;
            bestMove = acceptedMove;
            if(d > 1 && !sameMove(previousBest, bestMove)) bestMoveChanges++;
            ctx.stats.depthReached = d;
            ctx.stats.bestScore = bestScore;
            ctx.stats.bestMoveChanges = bestMoveChanges;
            ctx.stats.aspirationResearches = aspirationResearches;
            searchTT(ctx).store(bd.hash, d, scoreToTT(bestScore, 0), TTFlag::Exact, bestMove);

            const int scoreSwing = std::abs(bestScore - previousScore);
            int extraTime = 0;
            if(bestMoveChanges > 0){
                extraTime += std::min(softTimeLimitMs / 3, bestMoveChanges * std::max(20, softTimeLimitMs / 12));
            }
            if(aspirationResearches > 0){
                extraTime += std::min(softTimeLimitMs / 4, aspirationResearches * std::max(15, softTimeLimitMs / 20));
            }
            if(scoreSwing >= 80){
                extraTime += std::max(25, softTimeLimitMs / 8);
            }
            // Recompute from the original allocation, never a previous extension.
            dynamicSoftLimitMs = std::min(ctx.hardTimeLimitMs, softTimeLimitMs + extraTime);
            ctx.softTimeLimitMs = dynamicSoftLimitMs;
            ctx.stats.softTimeLimitMs = ctx.softTimeLimitMs;
        }
    }

    auto end = std::chrono::steady_clock::now();
    ctx.stats.timeMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - ctx.start).count();
    return bestMove;
}

inline Move searchBestMoveParallel(Board& bd, SearchContext& ctx, int maxDepth, int softTimeLimitMs, int hardTimeLimitMs,
                                   int threadCount, const std::vector<Move>* rootRestriction = nullptr,
                                   std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now()){
    ctx.stats = {};
    ctx.stats.softTimeLimitMs = softTimeLimitMs;
    ctx.stats.hardTimeLimitMs = hardTimeLimitMs;
    ctx.stats.configuredThreads = std::max(1, threadCount);
    ctx.stats.hardwareThreads = std::max(1, int(std::thread::hardware_concurrency()));
    ctx.start = started;
    ctx.softTimeLimitMs = softTimeLimitMs;
    ctx.hardTimeLimitMs = std::max(softTimeLimitMs, hardTimeLimitMs);
    ctx.stop = false;
    ctx.timeCheckCounter = 0;
    ctx.tt.newSearch();
    const Move noMove = invalidMove();

    for(int s=0; s<2; s++){
        for(int from=0; from<64; from++){
            for(int to=0; to<64; to++){
                ctx.history[s][from][to] = (ctx.history[s][from][to] * 7) / 8;
                ctx.continuationHistory[s][from][to] =
                    (ctx.continuationHistory[s][from][to] * 7) / 8;
            }
        }
        for(int pt=0; pt<7; pt++){
            for(int to=0; to<64; to++){
                ctx.captureHistory[s][pt][to] = (ctx.captureHistory[s][pt][to] * 7) / 8;
            }
        }
    }
    for(int ply = 0; ply < 128; ply++){
        ctx.plyMove[ply] = noMove;
        ctx.staticEvalByPly[ply] = -INF;
    }

    ctx.repetition = ctx.gameHistory;
    if(ctx.repetition.empty() || ctx.repetition.back() != bd.hash){
        ctx.repetition.push_back(bd.hash);
    }
    const std::vector<u64> rootRepetition = ctx.repetition;

    std::vector<Move> rootMoves;
    bd.genLegalMoves(rootMoves);
    if(rootRestriction){
        rootMoves.erase(std::remove_if(rootMoves.begin(), rootMoves.end(), [&](const Move& move){
            return std::none_of(rootRestriction->begin(), rootRestriction->end(), [&](const Move& allowed){
                return sameMove(move, allowed);
            });
        }), rootMoves.end());
    }
    if(rootMoves.empty()) return Move{};

    Move bestMove = rootMoves[0];
    int bestScore = 0;
    std::vector<int> rootScores(rootMoves.size(), 0);
    int bestMoveChanges = 0;
    int dynamicSoftLimitMs = ctx.softTimeLimitMs;

    const int workers = std::max(1, std::min<int>(std::clamp(threadCount, 1, 64), int(rootMoves.size())));
    ctx.stats.workersUsed = workers;
    std::vector<SearchContext> workerCtx;
    workerCtx.resize(static_cast<size_t>(workers));
    for(int w = 0; w < workers; w++){
        workerCtx[size_t(w)].sharedTT = &ctx.tt;
        workerCtx[size_t(w)].evaluator = ctx.evaluator;
        workerCtx[size_t(w)].incrementalNnue = ctx.incrementalNnue;
        refreshNnueRoot(bd, workerCtx[size_t(w)]);
        std::memcpy(workerCtx[size_t(w)].killer, ctx.killer, sizeof(ctx.killer));
        std::memcpy(workerCtx[size_t(w)].countermove, ctx.countermove, sizeof(ctx.countermove));
        std::memcpy(workerCtx[size_t(w)].history, ctx.history, sizeof(ctx.history));
        std::memcpy(workerCtx[size_t(w)].continuationHistory, ctx.continuationHistory,
                    sizeof(ctx.continuationHistory));
        std::memcpy(workerCtx[size_t(w)].captureHistory, ctx.captureHistory, sizeof(ctx.captureHistory));
    }

    int bestWorker = 0;
    RootWorkerPool workerPool(workers);

    for(int d=1; d<=maxDepth; d++){
        if(timeUp(ctx)) break;
        if(d > 1 && d >= 4 && softTimeUp(ctx)) break;

        std::vector<size_t> order(rootMoves.size());
        for(size_t i=0; i<rootMoves.size(); i++) order[i] = i;
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b){
            const bool aBest = sameMove(rootMoves[a], bestMove);
            const bool bBest = sameMove(rootMoves[b], bestMove);
            if(aBest != bBest) return aBest;
            if(rootScores[a] != rootScores[b]) return rootScores[a] > rootScores[b];
            return a < b;
        });

        std::vector<int> scores(rootMoves.size(), -INF);
        std::vector<int> owners(rootMoves.size(), -1);
        std::vector<u64> depthNodes(rootMoves.size(), 0);
        std::vector<u64> depthQNodes(rootMoves.size(), 0);

        for(int worker = 0; worker < workers; worker++){
            SearchContext& local = workerCtx[static_cast<size_t>(worker)];
            local.start = ctx.start;
            local.softTimeLimitMs = ctx.softTimeLimitMs;
            local.hardTimeLimitMs = ctx.hardTimeLimitMs;
            local.stop = false;
            local.timeCheckCounter = 0;
            local.abortFlag = ctx.abortFlag;
        }

        auto searchRootMove = [&](int worker, size_t index, int rootAlpha, bool fullWindow){
            SearchContext& local = workerCtx[static_cast<size_t>(worker)];
            Board child = bd;
            const Move move = rootMoves[index];
            Undo undo{};
            if(!child.makeMove(move, undo)) return;
            advanceNnue(child, undo, local, 0);

            local.repetition = rootRepetition;
            local.repetition.push_back(child.hash);
            const u64 previousNodes = local.stats.nodes;
            const u64 previousQNodes = local.stats.qnodes;

            int score = 0;
            if(fullWindow){
                score = -negamax(child, local, d - 1, -INF, INF, 1, move, true);
            } else {
                score = -negamax(child, local, d - 1, -rootAlpha - 1, -rootAlpha, 1, move, true);
                if(!local.stop && score > rootAlpha){
                    score = -negamax(child, local, d - 1, -INF, INF, 1, move, true);
                }
            }

            depthNodes[index] = local.stats.nodes - previousNodes;
            depthQNodes[index] = local.stats.qnodes - previousQNodes;
            if(local.stop) return;
            scores[index] = score;
            owners[index] = worker;
        };

        // Establish a trustworthy alpha before splitting the remaining root moves.
        const size_t firstIndex = order.front();
        searchRootMove(0, firstIndex, -INF, true);
        std::atomic<int> sharedAlpha{scores[firstIndex]};
        std::atomic<size_t> nextIndex{1};

        if(scores[firstIndex] != -INF){
            workerPool.run([&](int worker){
                while(true){
                    SearchContext& local = workerCtx[static_cast<size_t>(worker)];
                    if(timeUp(local)) break;
                    const size_t orderIndex = nextIndex.fetch_add(1, std::memory_order_relaxed);
                    if(orderIndex >= order.size()) break;

                    const size_t rootIndex = order[orderIndex];
                    const int alphaSnapshot = sharedAlpha.load(std::memory_order_relaxed);
                    searchRootMove(worker, rootIndex, alphaSnapshot, false);
                    const int score = scores[rootIndex];
                    int current = sharedAlpha.load(std::memory_order_relaxed);
                    while(score > current &&
                          !sharedAlpha.compare_exchange_weak(current, score, std::memory_order_relaxed)){}
                }
            });
        }

        bool incomplete = false;
        int localBest = -INF;
        size_t localBestIdx = 0;
        int localBestWorker = bestWorker;
        // Preserve principal-search order for equal scores. Iterating the raw
        // move-generation order can otherwise replace the exact first result
        // with a fail-low bound that merely happens to equal alpha.
        for(const size_t idx : order){
            if(scores[idx] == -INF){
                incomplete = true;
                continue;
            }
            rootScores[idx] = scores[idx];
            if(scores[idx] > localBest){
                localBest = scores[idx];
                localBestIdx = idx;
                localBestWorker = std::max(0, owners[idx]);
            }
            ctx.stats.nodes += depthNodes[idx];
            ctx.stats.qnodes += depthQNodes[idx];
        }

        auto now = std::chrono::steady_clock::now();
        const int elapsedMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.start).count();
        const bool externalAbort = ctx.abortFlag && ctx.abortFlag->load(std::memory_order_relaxed);
        const bool timedOut = elapsedMs >= ctx.hardTimeLimitMs;
        if((timedOut || externalAbort) && incomplete){
            ctx.stop = true;
            break;
        }

        if(localBest != -INF){
            const Move previousBest = bestMove;
            const int previousScore = bestScore;
            bestScore = localBest;
            bestMove = rootMoves[localBestIdx];
            bestWorker = localBestWorker;
            if(d > 1 && !sameMove(previousBest, bestMove)) bestMoveChanges++;
            ctx.stats.depthReached = d;
            ctx.stats.bestScore = bestScore;
            ctx.stats.bestMoveChanges = bestMoveChanges;
            ctx.tt.store(bd.hash, d, scoreToTT(bestScore, 0), TTFlag::Exact, bestMove);

            const int scoreSwing = std::abs(bestScore - previousScore);
            int extraTime = 0;
            if(bestMoveChanges > 0){
                extraTime += std::min(softTimeLimitMs / 3, bestMoveChanges * std::max(20, softTimeLimitMs / 12));
            }
            if(scoreSwing >= 80){
                extraTime += std::max(25, softTimeLimitMs / 8);
            }
            dynamicSoftLimitMs = std::min(ctx.hardTimeLimitMs, softTimeLimitMs + extraTime);
            ctx.softTimeLimitMs = dynamicSoftLimitMs;
            ctx.stats.softTimeLimitMs = ctx.softTimeLimitMs;
        }

        if(timedOut || externalAbort){
            ctx.stop = true;
            break;
        }
    }

    bestWorker = std::clamp(bestWorker, 0, workers - 1);
    std::memcpy(ctx.killer, workerCtx[size_t(bestWorker)].killer, sizeof(ctx.killer));
    std::memcpy(ctx.countermove, workerCtx[size_t(bestWorker)].countermove, sizeof(ctx.countermove));
    std::memcpy(ctx.history, workerCtx[size_t(bestWorker)].history, sizeof(ctx.history));
    std::memcpy(ctx.continuationHistory, workerCtx[size_t(bestWorker)].continuationHistory,
                sizeof(ctx.continuationHistory));
    std::memcpy(ctx.captureHistory, workerCtx[size_t(bestWorker)].captureHistory, sizeof(ctx.captureHistory));

    auto end = std::chrono::steady_clock::now();
    ctx.stats.timeMs = (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - ctx.start).count();
    return bestMove;
}

inline Move searchBestMove(Board& bd, SearchContext& ctx, int maxDepth, int softTimeLimitMs, int hardTimeLimitMs,
                           int threadCount=1, const std::vector<Move>* rootRestriction = nullptr,
                           std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now()){
    constexpr int MinimumParallelSearchMs = 100;
    const int requestedThreads = std::max(1, threadCount);
    if(requestedThreads <= 1 || hardTimeLimitMs < MinimumParallelSearchMs){
        Move best = searchBestMoveSingle(bd, ctx, maxDepth, softTimeLimitMs, hardTimeLimitMs, rootRestriction, started);
        ctx.stats.configuredThreads = requestedThreads;
        return best;
    }
    return searchBestMoveParallel(bd, ctx, maxDepth, softTimeLimitMs, hardTimeLimitMs, requestedThreads, rootRestriction, started);
}

inline std::string extractPVFromTT(Board bd, SearchContext& ctx, int maxPlies=12){
    std::string pv;
    std::vector<u64> seen;
    seen.reserve((size_t)maxPlies+2);

    for(int ply=0; ply<maxPlies; ply++){
        if(std::find(seen.begin(), seen.end(), bd.hash) != seen.end()) break;
        seen.push_back(bd.hash);

        const auto entry = ctx.tt.probe(bd.hash);
        if(!entry || entry->key != bd.hash) break;

        Move m = entry->best;

        MoveList leg;
        bd.genLegalMoves(leg);

        auto it = std::find_if(leg.begin(), leg.end(), [&](const Move& x){
            return x.from==m.from && x.to==m.to && x.promo==m.promo;
        });
        if(it == leg.end()) break;

        Undo u{};
        if(!bd.makeMove(*it, u)) break;

        if(!pv.empty()) pv += " ";
        pv += moveToUCI(*it);
    }
    return pv;
}

inline u64 perft(Board& bd, int depth){
    if(depth <= 0) return 1;

    MoveList legal;
    bd.genLegalMoves(legal);
    if(depth == 1) return static_cast<u64>(legal.size());

    u64 nodes = 0;
    for(const Move& m : legal){
        Undo u{};
        if(!bd.makeMove(m, u)) continue;
        nodes += perft(bd, depth - 1);
        bd.undoMove(u);
    }
    return nodes;
}

inline std::vector<std::pair<std::string, u64>> perftDivide(Board& bd, int depth){
    std::vector<std::pair<std::string, u64>> out;
    if(depth <= 0) return out;

    MoveList legal;
    bd.genLegalMoves(legal);
    out.reserve(legal.size());
    for(const Move& m : legal){
        Undo u{};
        if(!bd.makeMove(m, u)) continue;
        const u64 nodes = (depth == 1) ? 1 : perft(bd, depth - 1);
        bd.undoMove(u);
        out.emplace_back(moveToUCI(m), nodes);
    }
    return out;
}

struct PerftCase {
    const char* name;
    const char* fen;
    std::vector<std::pair<int, u64>> expectations;
};

inline int runPerftSuite(const Zobrist& zob, int maxDepthPerCase = 4){
    const std::vector<PerftCase> cases = {
        {"Start Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", {
            {1, 20ULL}, {2, 400ULL}, {3, 8902ULL}, {4, 197281ULL}
        }},
        {"Kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", {
            {1, 45ULL}, {2, 1947ULL}, {3, 85877ULL}, {4, 3617140ULL}
        }},
        {"Position 3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", {
            {1, 14ULL}, {2, 191ULL}, {3, 2812ULL}, {4, 43238ULL}
        }},
        {"Position 4", "r3k2r/Pppp1ppp/1b3nbN/nP6/B1P1P3/5N2/Pp1P1PPP/R2Q1RK1 w kq - 0 1", {
            {1, 30ULL}, {2, 1160ULL}, {3, 35941ULL}, {4, 1371859ULL}
        }},
        {"Position 5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", {
            {1, 44ULL}, {2, 1486ULL}, {3, 62379ULL}, {4, 2103487ULL}
        }},
        {"Position 6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 10", {
            {1, 44ULL}, {2, 1987ULL}, {3, 83034ULL}, {4, 3596057ULL}
        }}
    };

    int failures = 0;
    for(const auto& tc : cases){
        Board bd;
        bd.setZobrist(&zob);
        if(!bd.loadFEN(tc.fen)){
            std::cout << "[FAIL] " << tc.name << " (invalid FEN)\n";
            failures++;
            continue;
        }

        std::cout << "\n== " << tc.name << " ==\n";
        for(const auto& [depth, expected] : tc.expectations){
            if(depth > maxDepthPerCase) continue;
            auto t0 = std::chrono::steady_clock::now();
            const u64 got = perft(bd, depth);
            auto t1 = std::chrono::steady_clock::now();
            const int ms = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            const double nps = (ms > 0) ? (double(got) * 1000.0 / double(ms)) : 0.0;

            const bool ok = (got == expected);
            std::cout << "d" << depth
                      << " expected=" << expected
                      << " got=" << got
                      << " time=" << ms << "ms"
                      << " nps=" << static_cast<long long>(nps)
                      << (ok ? " [OK]" : " [FAIL]")
                      << "\n";
            if(!ok) failures++;
        }
    }

    if(failures == 0){
        std::cout << "\nPerft suite: PASS\n";
        return 0;
    }
    std::cout << "\nPerft suite: FAIL (" << failures << " mismatches)\n";
    return 1;
}

struct BenchmarkPosition {
    const char* name;
    const char* fen;
};

inline int runSearchBenchmark(const Zobrist& zob, int depth, int perPositionTimeMs,
                              int ttSizeMB = 256, int threads = 1,
                              const PositionEvaluator* evaluator = nullptr,
                              bool incrementalNnue = true){
    const std::vector<BenchmarkPosition> positions = {
        {"Start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"},
        {"Middlegame 1", "r2q1rk1/pp2bppp/2np1n2/2p1p1B1/2P1P3/2NP1N2/PP2QPPP/R4RK1 w - - 0 10"},
        {"Middlegame 2", "r1bq1rk1/pp2bppp/2n1pn2/2pp4/2P5/2NP1NP1/PP2PPBP/R1BQ1RK1 w - - 0 9"},
        {"Endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"}
    };

    u64 totalNodes = 0;
    int totalMs = 0;
    std::cout << "Benchmark: depth=" << depth
              << " timeLimit=" << perPositionTimeMs
              << "ms threads=" << threads
              << " evaluator=" << (evaluator ? evaluator->backendName() : "classical")
              << " nnueWeight=" << (evaluator ? evaluator->nnueWeight() : 0)
              << " nnueMode=" << (evaluator && evaluator->usingNnue()
                    ? (incrementalNnue ? "incremental" : "rebuild") : "n/a")
              << " positions=" << positions.size() << "\n";

    for(const auto& p : positions){
        Board bd;
        bd.setZobrist(&zob);
        if(!bd.loadFEN(p.fen)){
            std::cout << "[FAIL] " << p.name << " invalid FEN\n";
            return 1;
        }

        SearchContext ctx;
        ctx.tt.resizeMB(static_cast<size_t>(ttSizeMB));
        ctx.gameHistory = {bd.hash};
        ctx.evaluator = evaluator;
        ctx.incrementalNnue = incrementalNnue;

        const Move best = searchBestMove(bd, ctx, depth, perPositionTimeMs, perPositionTimeMs, threads);
        const double nps = (ctx.stats.timeMs > 0)
            ? (double(ctx.stats.nodes) * 1000.0 / double(ctx.stats.timeMs))
            : 0.0;

        std::cout << std::left << std::setw(12) << p.name
                  << " best=" << moveToUCI(best)
                  << " depth=" << ctx.stats.depthReached
                  << " score=" << ctx.stats.bestScore
                  << " nodes=" << ctx.stats.nodes
                  << " qnodes=" << ctx.stats.qnodes
                  << " time=" << ctx.stats.timeMs << "ms"
                  << " nps=" << static_cast<long long>(nps)
                  << "\n";

        totalNodes += ctx.stats.nodes;
        totalMs += ctx.stats.timeMs;
    }

    const double totalNps = (totalMs > 0) ? (double(totalNodes) * 1000.0 / double(totalMs)) : 0.0;
    std::cout << "Benchmark summary: nodes=" << totalNodes
              << " time=" << totalMs << "ms"
              << " nps=" << static_cast<long long>(totalNps)
              << "\n";
    return 0;
}
