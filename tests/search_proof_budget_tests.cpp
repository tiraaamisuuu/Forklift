#include "chess_core.hpp"

namespace {
int failures = 0;
void expect(bool value, const char* message){
    if(!value){ std::cerr << "[FAIL] " << message << '\n'; failures++; }
}

void testQuota(){
    auto owner = std::make_unique<SearchContext>();
    SearchContext& context = *owner;
    context.singularProofBudgetPercent = 10;
    expect(singularProofQuota(context) == 256 && singularProofBudgetAvailable(context),
           "the bootstrap admits initial verification without ordinary work");
    context.stats.nodes = 256;
    context.stats.singularProofNodes = 256;
    expect(!singularProofBudgetAvailable(context), "an exactly spent quota has no remaining headroom");
    context.stats.nodes = 1000;
    context.stats.qnodes = 1000;
    context.stats.singularProofNodes = 300;
    expect(singularProofQuota(context) == 426 && singularProofBudgetAvailable(context),
           "only non-proof main and quiescence work earn quota");
    context.stats.singularProofNodes = 500;
    expect(singularProofQuota(context) == 406 && !singularProofBudgetAvailable(context),
           "overspent proof work must block further admissions");
    context.stats.nodes = 0;
    context.stats.qnodes = 1000;
    context.stats.singularProofNodes = 1000;
    expect(singularProofQuota(context) == 256 && !singularProofBudgetAvailable(context),
           "proof work cannot fund its own next proof");
    context.singularProofBudgetPercent = 0;
    expect(singularProofBudgetAvailable(context), "zero budget preserves unrestricted legacy behavior");
    context.singularProofBudgetPercent = 100;
    context.stats.nodes = ~u64{0};
    context.stats.qnodes = 100;
    context.stats.singularProofNodes = 0;
    expect(singularProofQuota(context) == ~u64{0}, "large counters must saturate, not wrap");
    context.stats.singularProofNodes = ~u64{0};
    expect(singularProofQuota(context) == 256 && !singularProofBudgetAvailable(context),
           "large proof debt must not wrap into a free admission");
}

void testLiveLedger(){
    const Zobrist zobrist;
    Board board;
    board.setZobrist(&zobrist);
    expect(board.loadFEN("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"), "ledger fixture loads");
    const std::string original = board.toFEN();
    const u64 originalHash = board.hash;
    MoveList legal;
    board.genLegalMoves(legal);
    Move capture = invalidMove();
    for(const Move& move : legal) if(moveToUCI(move) == "d2d5") capture = move;
    expect(capture.from < 64, "ledger proving capture is legal");
    for(const bool block : {false, true}){
        auto context = std::make_unique<SearchContext>();
        context->tt.resizeMB(1);
        context->start = std::chrono::steady_clock::now();
        context->softTimeLimitMs = 60'000;
        context->hardTimeLimitMs = 60'000;
        context->repetition = {board.hash};
        context->enableSingularExtensions = true;
        context->singularProofBudgetPercent = block ? 1 : 0;
        if(block){ // Inject known debt without relying on wall-clock timing.
            context->stats.nodes = 1'000'000'000;
            context->stats.singularProofNodes = 1'000'000'000;
        }
        context->tt.store(board.hash, 5, 450, TTFlag::Lower, capture);
        (void)negamax(board, *context, 8, 399, 400, 1, invalidMove(), false, false);
        if(block){
            expect(context->stats.singularBudgetSkips > 0 && context->stats.singularAttempts == 0,
                   "an overspent live node skips rather than attempting verification");
        } else {
            expect(context->stats.singularAttempts > 0 && context->stats.singularProofNodes > 0 &&
                   context->stats.singularProofNodes <= context->stats.nodes + context->stats.qnodes,
                   "unrestricted proofs charge real work once, including quiescence");
        }
        expect(board.toFEN() == original && board.hash == originalHash &&
               context->repetition == std::vector<u64>{originalHash} && context->activeScratch == 0 &&
               !context->singularVerification && context->activeSingularExtensions == 0,
               "budget admission restores board, repetition and all temporary search state");
    }
}

void testInterruptedRootWork(){
    SearchStats aggregate;
    SearchStats interrupted;
    interrupted.nodes = 127;
    interrupted.qnodes = 63;
    interrupted.singularAttempts = 2;
    interrupted.singularExtensions = 1;
    interrupted.singularProofNodes = 30;
    interrupted.singularBudgetSkips = 4;
    interrupted.maxSingularExtensionsOnPath = 1;
    accumulateSearchWork(aggregate, interrupted);
    expect(aggregate.nodes == 127 && aggregate.qnodes == 63 &&
           aggregate.singularAttempts == 2 && aggregate.singularExtensions == 1 &&
           aggregate.singularProofNodes == 30 && aggregate.singularBudgetSkips == 4 &&
           aggregate.maxSingularExtensionsOnPath == 1,
           "partial root work must be accounted independently of a valid score");
    accumulateSearchWork(aggregate, SearchStats{});
    expect(aggregate.singularProofNodes == 30 && aggregate.maxSingularExtensionsOnPath == 1,
           "unstarted moves must add no work and cannot clear a recorded peak");
}
} // namespace

int main(){
    testQuota();
    testLiveLedger();
    testInterruptedRootWork();
    if(failures) return 1;
    std::cout << "Proof budget tests: PASS\n";
    return 0;
}
