#include "chess_core.hpp"

namespace {

int failures = 0;

void expect(bool condition, const std::string& message){
    if(condition) return;
    std::cerr << "[FAIL] " << message << '\n';
    failures++;
}

bool samePosition(const Board& lhs, const Board& rhs){
    if(lhs.stm != rhs.stm || lhs.epSquare != rhs.epSquare || lhs.castling != rhs.castling ||
       lhs.halfmoveClock != rhs.halfmoveClock || lhs.fullmoveNumber != rhs.fullmoveNumber ||
       lhs.kingSquare != rhs.kingSquare || lhs.hash != rhs.hash) return false;
    for(size_t index = 0; index < lhs.b.size(); index++){
        if(lhs.b[index].t != rhs.b[index].t || lhs.b[index].c != rhs.b[index].c) return false;
    }
    return true;
}

Move findMove(Board& board, const std::string& notation){
    MoveList legal;
    board.genLegalMoves(legal);
    for(const Move& move : legal){
        if(moveToUCI(move) == notation) return move;
    }
    return invalidMove();
}

std::unique_ptr<SearchContext> makeContext(const Board& board){
    auto context = std::make_unique<SearchContext>();
    context->tt.resizeMB(4);
    context->enableSingularExtensions = true;
    context->singularProofBudgetPercent = 0;
    context->repetition = {board.hash};
    context->gameHistory = context->repetition;
    context->softTimeLimitMs = 60'000;
    context->hardTimeLimitMs = 60'000;
    context->start = std::chrono::steady_clock::now();
    context->plyMove[1] = invalidMove();
    return context;
}

void expireAtPeriodicCheck(SearchContext& context, u32 firstCounter = 1){
    // timeUp checks a 60-second budget every 256 invocations. A nonzero
    // starting counter skips entry, then expires at the next periodic check.
    // This is deterministic: no sleeping, timer thread, or scheduling race.
    context.start = std::chrono::steady_clock::now() - std::chrono::hours(1);
    context.timeCheckCounter = firstCounter;
}

void checkUnwound(const Board& board, const Board& original, const SearchContext& context,
                 const std::vector<u64>& repetition, const std::string& label){
    expect(context.stop, label + " must observe the expired deadline during recursion");
    expect(context.stats.nodes > 1,
           label + " must do recursive work before cancellation");
    expect(context.stats.singularAttempts > 0 && context.stats.singularProofNodes > 0,
           label + " must actually enter singular verification before cancellation");
    expect(context.stats.singularProofNodes <= context.stats.nodes + context.stats.qnodes,
           label + " proof ledger must not exceed all recorded search work");
    expect(samePosition(board, original), label + " must restore every board field");
    expect(context.repetition == repetition, label + " must restore repetition history");
    expect(context.activeScratch == 0, label + " must release every scratch frame");
    expect(!context.singularVerification && context.activeSingularExtensions == 0,
           label + " must unwind verification and extension modes");
    expect(context.stats.maxSingularExtensionsOnPath <= SearchContext::MaxSingularExtensionsOnPath,
           label + " must respect the path extension cap");
}

void testInFlightVerificationCancellation(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    expect(board.loadFEN("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"),
           "verification-cancellation fixture must load");
    const Board original = board;
    auto context = makeContext(board);
    const auto repetition = context->repetition;
    const Move candidate = findMove(board, "d2d5");
    expect(candidate.from < 64, "verification candidate must be legal");
    context->tt.store(board.hash, 5, 450, TTFlag::Lower, candidate);
    const auto rootEntry = context->tt.probe(board.hash);
    expireAtPeriodicCheck(*context, 240);
    (void)negamax(board, *context, 8, 399, 400, 1, invalidMove(), false, false);
    checkUnwound(board, original, *context, repetition, "in-flight verification");
    expect(context->stats.singularExtensions == 0,
           "verification fixture must stop before admitting an extension");
    expect(context->plyMove[1].from >= 64,
           "stopped same-ply verification must restore the parent's current move");
    expect(context->staticEvalByPly[1] == evaluateClassical(board),
           "stopped same-ply verification must preserve the parent's static evaluation");
    const auto afterEntry = context->tt.probe(board.hash);
    expect(rootEntry && afterEntry && rootEntry->depth == afterEntry->depth &&
           rootEntry->score == afterEntry->score && rootEntry->flag == afterEntry->flag &&
           sameMove(rootEntry->best, afterEntry->best),
           "stopped verification must not publish a restricted result to the root TT");
}

void testInFlightExtensionCancellation(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    expect(board.loadFEN("7k/8/6K1/8/8/8/8/R7 b - - 0 1"),
           "extension-cancellation fixture must load");
    MoveList legal;
    board.genLegalMoves(legal);
    expect(legal.size() == 1 && !board.inCheck(board.stm),
           "extension fixture must have one legal move without being in check");
    if(legal.size() != 1) return;
    const Board original = board;
    auto context = makeContext(board);
    const auto repetition = context->repetition;
    context->tt.store(board.hash, 5, 0, TTFlag::Lower, legal[0]);
    expireAtPeriodicCheck(*context);
    (void)negamax(board, *context, 8, -INF, INF, 1, invalidMove(), false, false);
    checkUnwound(board, original, *context, repetition, "in-flight extended branch");
    expect(context->stats.singularExtensions > 0 && context->stats.maxSingularExtensionsOnPath > 0,
           "extension cancellation must happen after an extension scope becomes active");
}

void testParallelSingularActivation(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    expect(board.loadFEN("r2q1rk1/pp2bppp/2np1n2/2p1p1B1/2P1P3/2NP1N2/PP2QPPP/R4RK1 w - - 0 10"),
           "parallel singular fixture must load");
    const Board original = board;
    auto context = makeContext(board);
    context->tt.resizeMB(16);
    const auto repetition = context->repetition;
    MoveList legal;
    board.genLegalMoves(legal);
    const Move best = searchBestMove(board, *context, 10, 60'000, 60'000, 4);
    expect(std::any_of(legal.begin(), legal.end(), [&](const Move& move){ return sameMove(move, best); }),
           "parallel singular search must return a legal root move");
    expect(context->stats.depthReached == 10 && !context->stop,
           "parallel singular fixture must complete its prescribed depth");
    expect(context->stats.configuredThreads == 4 && context->stats.workersUsed == 4,
           "parallel singular fixture must actually activate four workers");
    expect(context->stats.singularAttempts > 0 && context->stats.singularProofNodes > 0,
           "parallel workers must propagate opt-in singular verification and its work counters");
    expect(context->stats.singularExtensions > 0 && context->stats.maxSingularExtensionsOnPath > 0,
           "parallel workers must exercise extension scopes, not only unsuccessful proofs");
    expect(context->stats.singularProofNodes <= context->stats.nodes + context->stats.qnodes,
           "parallel proof accounting must stay within all aggregated search work");
    expect(context->stats.maxSingularExtensionsOnPath <= SearchContext::MaxSingularExtensionsOnPath,
           "parallel workers must respect the singular path cap");
    expect(samePosition(board, original) && context->repetition == repetition,
           "parallel singular search must preserve the root board and repetition history");
    expect(context->activeScratch == 0 && !context->singularVerification &&
           context->activeSingularExtensions == 0,
           "parallel singular search must leave no temporary mode active in the root context");
}

} // namespace

int main(){
    const Zobrist zobrist;
    testInFlightVerificationCancellation(zobrist);
    testInFlightExtensionCancellation(zobrist);
    testParallelSingularActivation(zobrist);
    if(failures != 0){
        std::cerr << failures << " singular-search test assertion(s) failed\n";
        return 1;
    }
    std::cout << "Singular search cancellation and parallel tests: PASS\n";
    return 0;
}
