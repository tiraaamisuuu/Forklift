#include "chess_core.hpp"
#include "time_management.hpp"

#include <functional>
#include <random>

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
       lhs.kingSquare != rhs.kingSquare || lhs.hash != rhs.hash){
        return false;
    }
    for(size_t index = 0; index < lhs.b.size(); index++){
        if(lhs.b[index].t != rhs.b[index].t || lhs.b[index].c != rhs.b[index].c) return false;
    }
    return true;
}

Move findMove(Board& board, const std::string& uci){
    std::vector<Move> legal;
    board.genLegalMoves(legal);
    const auto found = std::find_if(legal.begin(), legal.end(), [&](const Move& move){
        return moveToUCI(move) == uci;
    });
    if(found == legal.end()) return invalidMove();
    return *found;
}

void testPerft(const Zobrist& zobrist, bool quick){
    struct Case { const char* fen; int depth; u64 expected; };
    const std::vector<Case> cases = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         quick ? 3 : 4, quick ? 8902ULL : 197281ULL},
        {"r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", 3, 85877ULL},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812ULL},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/B1P1P3/5N2/Pp1P1PPP/R2Q1RK1 w kq - 0 1", 3, 35941ULL},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379ULL},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 10", 3, 83034ULL}
    };

    for(const Case& test : cases){
        Board board;
        board.setZobrist(&zobrist);
        expect(board.loadFEN(test.fen), "perft FEN should load");
        expect(perft(board, test.depth) == test.expected, "perft mismatch at depth " + std::to_string(test.depth));
    }
}

void testMakeUnmakeAndHash(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    const Board initial = board;

    std::vector<Move> legal;
    board.genLegalMoves(legal);
    for(const Move& move : legal){
        Undo undo{};
        expect(board.makeMove(move, undo), "generated legal move should be makeable");
        const u64 incrementalHash = board.hash;
        board.recomputeHash();
        expect(board.hash == incrementalHash, "incremental hash should match recomputed hash");
        board.undoMove(undo);
        expect(samePosition(board, initial), "make/unmake should restore the exact root position");
    }

    std::mt19937 random(0x54495241U);
    std::vector<Undo> undoStack;
    for(int ply = 0; ply < 160; ply++){
        board.genLegalMoves(legal);
        if(legal.empty()) break;
        const Move move = legal[static_cast<size_t>(random() % legal.size())];
        Undo undo{};
        expect(board.makeMove(move, undo), "random legal move should be makeable");
        undoStack.push_back(undo);
        const u64 incrementalHash = board.hash;
        board.recomputeHash();
        expect(board.hash == incrementalHash, "random playout hash should remain incremental");
    }
    while(!undoStack.empty()){
        board.undoMove(undoStack.back());
        undoStack.pop_back();
    }
    expect(samePosition(board, initial), "random playout should unwind exactly to the start position");
}

void testNullMoveAndHash(const Zobrist& zobrist){
    const std::array<const char*, 3> positions{{
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 4 7",
        "rnbqkbnr/pppp1ppp/8/8/4pP2/8/PPPP2PP/RNBQKBNR b KQkq f3 9 12",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 37 61"
    }};

    for(const char* fen : positions){
        Board board;
        board.setZobrist(&zobrist);
        expect(board.loadFEN(fen), "null-move FEN should load");
        const Board original = board;
        const Color originalSide = board.stm;
        const int originalHalfmove = board.halfmoveClock;
        const int originalFullmove = board.fullmoveNumber;

        NullUndo undo{};
        board.makeNullMove(undo);
        expect(board.stm == other(originalSide), "null move should change side to move");
        expect(board.epSquare == -1, "null move should clear en passant");
        expect(board.halfmoveClock == originalHalfmove && board.fullmoveNumber == originalFullmove,
               "search-only null move should preserve move counters");
        const u64 incrementalHash = board.hash;
        board.recomputeHash();
        expect(board.hash == incrementalHash, "null-move incremental hash should match recomputation");

        board.undoNullMove(undo);
        expect(samePosition(board, original), "null move should restore the exact position");
    }
}

void testEvaluationOrientation(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    board.reset();

    const int e2 = sqToIndex(Square{4, 1});
    const int e4 = sqToIndex(Square{4, 3});
    expect(PST_PAWN[mirrorIndex(e4)] > PST_PAWN[mirrorIndex(e2)],
           "white pawn PST should reward e4 over its initial e2 square");

    const Move e2e4 = findMove(board, "e2e4");
    expect(e2e4.from < 64, "e2e4 should be legal from the start position");
}

void testPseudoMobilityCounter(const Zobrist& zobrist){
    auto compare = [&](const Board& position, const std::string& label){
        for(const Color color : {Color::White, Color::Black}){
            Board generated = position;
            generated.stm = color;
            std::vector<Move> moves;
            generated.genPseudoMoves(moves);
            expect(pseudoMobility(position, color) == static_cast<int>(moves.size()),
                   label + " mobility counter should match generated pseudo moves");
        }
    };

    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    compare(board, "start position");

    const std::vector<std::string> fixtures = {
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/P6p/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/2q5/3r4/2B1N3/3Q4/8/4R3/4K3 b - - 0 1",
    };
    for(const std::string& fen : fixtures){
        expect(board.loadFEN(fen), "mobility fixture should load");
        compare(board, "fixture");
    }

    board.reset();
    std::mt19937 random(0x4D4F4249U);
    for(int ply = 0; ply < 100; ply++){
        compare(board, "random playout");
        std::vector<Move> legal;
        board.genLegalMoves(legal);
        if(legal.empty()) break;
        Undo undo{};
        expect(board.makeMove(legal[static_cast<size_t>(random() % legal.size())], undo),
               "random mobility move should be makeable");
    }
}

void testEnPassantHashing(const Zobrist& zobrist){
    Board unavailableEp;
    unavailableEp.setZobrist(&zobrist);
    expect(unavailableEp.loadFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
           "unavailable en-passant FEN should load");
    Board noEp;
    noEp.setZobrist(&zobrist);
    expect(noEp.loadFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
           "matching no-en-passant FEN should load");
    expect(unavailableEp.hash == noEp.hash,
           "unavailable en-passant target must not alter repetition hash");

    Board availableEp;
    availableEp.setZobrist(&zobrist);
    expect(availableEp.loadFEN("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"),
           "available en-passant FEN should load");
    Board availableNoEp;
    availableNoEp.setZobrist(&zobrist);
    expect(availableNoEp.loadFEN("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1"),
           "available position without en-passant target should load");
    expect(availableEp.hash != availableNoEp.hash,
           "available en-passant capture must alter repetition hash");
}

void testDrawRules(const Zobrist& zobrist){
    Board sameColorBishops;
    sameColorBishops.setZobrist(&zobrist);
    expect(sameColorBishops.loadFEN("5b1k/8/8/8/8/8/8/K1B5 w - - 0 1"), "same-colour bishop FEN should load");
    expect(sameColorBishops.insufficientMaterial(), "same-colour K+B vs K+B should be dead");

    Board oppositeColorBishops;
    oppositeColorBishops.setZobrist(&zobrist);
    expect(oppositeColorBishops.loadFEN("6bk/8/8/8/8/8/8/K1B5 w - - 0 1"), "opposite-colour bishop FEN should load");
    expect(!oppositeColorBishops.insufficientMaterial(), "opposite-colour K+B vs K+B is not automatically dead");

    Board repeated;
    repeated.setZobrist(&zobrist);
    repeated.reset();
    const std::vector<u64> history{repeated.hash, repeated.hash, repeated.hash};
    expect(assessGameStatus(repeated, history).termination == GameTermination::ThreefoldRepetition,
           "third occurrence should adjudicate a repetition draw");

    Board checkmateAtHundred;
    checkmateAtHundred.setZobrist(&zobrist);
    expect(checkmateAtHundred.loadFEN("7k/6Q1/6K1/8/8/8/8/8 b - - 100 75"), "checkmate FEN should load");
    expect(assessGameStatus(checkmateAtHundred, {checkmateAtHundred.hash}).termination == GameTermination::BlackCheckmated,
           "checkmate must take precedence over a rule-draw claim");
}

void testNotation(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    Move move = findMove(board, "e2e4");
    expect(moveToSAN(board, move) == "e4", "e2e4 SAN should be e4");

    Undo undo{};
    expect(board.makeMove(move, undo), "e2e4 should be makeable for SAN continuation");
    move = findMove(board, "e7e5");
    expect(moveToSAN(board, move) == "e5", "e7e5 SAN should be e5");
}

void testFENValidation(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    expect(!board.loadFEN("8/8/8/8/8/8/8 w - - 0 1"), "FEN with seven ranks must fail");
    expect(!board.loadFEN("8/8/8/8/8/8/8/9 w - - 0 1"), "FEN digit 9 must fail");
    expect(!board.loadFEN("8/8/8/8/8/8/8/8 x - - 0 1"), "invalid side to move must fail");
    expect(!board.loadFEN("8/8/8/8/8/8/8/8 w - - 0 1"), "kingless FEN must fail");

    const std::string fen = "r3k2r/ppp2ppp/2n5/3qp3/8/2N2N2/PPP2PPP/R2Q1RK1 b kq - 7 14";
    expect(board.loadFEN(fen), "round-trip FEN should load");
    expect(board.toFEN() == fen, "FEN serialization should preserve all six fields");
}

void testTranspositionClusters(){
    TranspositionTable table;
    table.resizeMB(1);
    const u64 stride = static_cast<u64>(table.mask + 1);
    const Move move{};
    for(u64 collision = 0; collision < TranspositionTable::ClusterSize; collision++){
        const u64 key = 17 + collision * stride;
        table.store(key, static_cast<int>(collision + 1), static_cast<int>(collision * 10), TTFlag::Exact, move);
    }
    for(u64 collision = 0; collision < TranspositionTable::ClusterSize; collision++){
        const u64 key = 17 + collision * stride;
        const auto entry = table.probe(key);
        expect(entry && entry->key == key, "TT cluster should retain colliding positions");
    }

    Move packedMove{};
    packedMove.from = 12;
    packedMove.to = 34;
    packedMove.promo = PieceType::Queen;
    table.store(0x123456789ABCDEF0ULL, 63, -12345, TTFlag::Upper, packedMove);
    const auto packedEntry = table.probe(0x123456789ABCDEF0ULL);
    expect(packedEntry && packedEntry->depth == 63 && packedEntry->score == -12345 &&
           packedEntry->flag == TTFlag::Upper &&
           packedEntry->best.from == 12 && packedEntry->best.to == 34 &&
           packedEntry->best.promo == PieceType::Queen,
           "packed TT entry should preserve score, depth, flag, and best move");
}

void testConcurrentTranspositionTable(){
    TranspositionTable table;
    table.resizeMB(1);
    const u64 stride = static_cast<u64>(table.mask + 1);
    constexpr int WorkerCount = 8;
    constexpr int StoresPerWorker = 2000;
    std::atomic<bool> valid{true};
    std::vector<std::thread> workers;
    workers.reserve(WorkerCount);

    for(int worker = 0; worker < WorkerCount; worker++){
        workers.emplace_back([&, worker](){
            for(int index = 1; index <= StoresPerWorker; index++){
                const u64 serial = static_cast<u64>(worker * StoresPerWorker + index);
                const u64 key = 29 + serial * stride;
                const int depth = 1 + int(serial % 63);
                const int score = int(serial % 60001) - 30000;
                Move move{};
                move.from = static_cast<u8>(serial % 64);
                move.to = static_cast<u8>((serial / 3) % 64);
                move.promo = static_cast<PieceType>(serial % 7);
                const TTFlag flag = static_cast<TTFlag>(serial % 3);
                table.store(key, depth, score, flag, move);

                const auto entry = table.probe(key);
                if(entry && (entry->depth != depth || entry->score != score ||
                   entry->flag != flag || entry->best.from != move.from ||
                   entry->best.to != move.to || entry->best.promo != move.promo)){
                    valid.store(false, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }

    for(std::thread& worker : workers) worker.join();
    expect(valid.load(std::memory_order_relaxed),
           "concurrent TT collisions must produce complete entries or clean misses");
}

void testNnueFormat(const Zobrist& zobrist){
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "chess-engine-nnue-format-test.nnue";
    {
        std::ofstream output(path, std::ios::binary);
        const std::array<char, 8> magic{{'T','N','N','U','E','1','\0','\0'}};
        const u32 version = NnueNetwork::FormatVersion;
        const u32 features = NnueNetwork::FeatureCount;
        const u32 hidden = 1;
        const int32_t hiddenScale = 1;
        const int32_t outputScale = 1;
        const int32_t outputBias = 42;
        const int32_t hiddenBias = 0;
        const std::vector<int16_t> inputWeights(features, 0);
        const std::array<int16_t, 2> outputWeights{{0, 0}};
        output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        output.write(reinterpret_cast<const char*>(&version), sizeof(version));
        output.write(reinterpret_cast<const char*>(&features), sizeof(features));
        output.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
        output.write(reinterpret_cast<const char*>(&hiddenScale), sizeof(hiddenScale));
        output.write(reinterpret_cast<const char*>(&outputScale), sizeof(outputScale));
        output.write(reinterpret_cast<const char*>(&outputBias), sizeof(outputBias));
        output.write(reinterpret_cast<const char*>(&hiddenBias), sizeof(hiddenBias));
        output.write(reinterpret_cast<const char*>(inputWeights.data()),
                     static_cast<std::streamsize>(inputWeights.size() * sizeof(int16_t)));
        output.write(reinterpret_cast<const char*>(outputWeights.data()), sizeof(outputWeights));
    }

    PositionEvaluator evaluator;
    std::string error;
    expect(evaluator.loadNnue(path, &error), "valid NNUE v1 fixture should load: " + error);
    expect(evaluator.setUseNnue(true), "loaded NNUE should be selectable");
    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    expect(evaluator.evaluate(board) == 42, "NNUE quantized inference should match fixture output");

    const int classical = evaluateClassical(board);
    evaluator.setNnueWeight(25);
    const int expectedHybrid = (classical * 75 + 42 * 25) / 100;
    expect(evaluator.evaluate(board) == expectedHybrid,
           "hybrid evaluation should use the configured NNUE percentage");
    NnueAccumulator accumulator;
    evaluator.refreshAccumulator(board, accumulator);
    expect(evaluator.evaluate(board, accumulator) == expectedHybrid,
           "incremental hybrid evaluation should match full evaluation");
    expect(std::string(evaluator.backendName()) == "hybrid",
           "partial NNUE weight should report the hybrid backend");

    evaluator.setNnueWeight(-1);
    expect(evaluator.nnueWeight() == 0 && evaluator.evaluate(board) == classical,
           "NNUE weight should clamp to the classical endpoint");
    evaluator.setNnueWeight(101);
    expect(evaluator.nnueWeight() == 100 && evaluator.evaluate(board) == 42,
           "NNUE weight should clamp to the neural endpoint");
    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

void testIncrementalNnue(const Zobrist& zobrist){
    const std::filesystem::path path = std::filesystem::temp_directory_path() /
        "chess-engine-nnue-incremental-test.nnue";
    {
        std::ofstream output(path, std::ios::binary);
        const std::array<char, 8> magic{{'T','N','N','U','E','1','\0','\0'}};
        const u32 version = NnueNetwork::FormatVersion;
        const u32 features = NnueNetwork::FeatureCount;
        const u32 hidden = 4;
        const int32_t hiddenScale = 16;
        const int32_t outputScale = 4;
        const int32_t outputBias = 11;
        const std::array<int32_t, 4> hiddenBias{{2, 3, 4, 5}};
        std::vector<int16_t> inputWeights(static_cast<size_t>(features) * hidden);
        for(size_t feature = 0; feature < features; feature++){
            for(size_t unit = 0; unit < hidden; unit++){
                inputWeights[feature * hidden + unit] = static_cast<int16_t>(
                    (feature * 13 + unit * 7) % 7 - 3);
            }
        }
        const std::array<int16_t, 8> outputWeights{{3, -2, 5, -4, -3, 6, -2, 5}};
        output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        output.write(reinterpret_cast<const char*>(&version), sizeof(version));
        output.write(reinterpret_cast<const char*>(&features), sizeof(features));
        output.write(reinterpret_cast<const char*>(&hidden), sizeof(hidden));
        output.write(reinterpret_cast<const char*>(&hiddenScale), sizeof(hiddenScale));
        output.write(reinterpret_cast<const char*>(&outputScale), sizeof(outputScale));
        output.write(reinterpret_cast<const char*>(&outputBias), sizeof(outputBias));
        output.write(reinterpret_cast<const char*>(hiddenBias.data()), sizeof(hiddenBias));
        output.write(reinterpret_cast<const char*>(inputWeights.data()),
                     static_cast<std::streamsize>(inputWeights.size() * sizeof(int16_t)));
        output.write(reinterpret_cast<const char*>(outputWeights.data()), sizeof(outputWeights));
    }

    NnueNetwork network;
    std::string error;
    expect(network.load(path, &error), "incremental NNUE fixture should load: " + error);

    auto compare = [&](const Board& board, const NnueAccumulator& incremental,
                       const std::string& label){
        NnueAccumulator rebuilt;
        network.refresh(board, rebuilt);
        expect(incremental.valid && rebuilt.valid, label + " accumulators should be valid");
        expect(incremental.values == rebuilt.values,
               label + " incremental accumulator should equal a full rebuild");
        expect(network.evaluate(board, incremental) == network.evaluate(board),
               label + " incremental evaluation should equal reference inference");
    };

    auto exerciseMove = [&](const std::string& fen, const std::string& uci,
                            const std::string& label){
        Board board;
        board.setZobrist(&zobrist);
        expect(board.loadFEN(fen), label + " FEN should load");
        NnueAccumulator parent;
        network.refresh(board, parent);
        const Move move = findMove(board, uci);
        expect(move.from < 64, label + " move should be legal");
        Undo undo{};
        if(move.from < 64 && board.makeMove(move, undo)){
            NnueAccumulator child;
            network.updateAfterMove(board, undo, parent, child);
            compare(board, child, label);
            board.undoMove(undo);
            compare(board, parent, label + " after unmake");
        }
    };

    exerciseMove("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", "castling");
    exerciseMove("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", "en passant");
    exerciseMove("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "promotion");
    exerciseMove("4k3/8/8/8/8/3p4/4K3/8 w - - 0 1", "e2d3", "king capture");

    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    NnueAccumulator accumulator;
    network.refresh(board, accumulator);
    std::vector<Undo> undos;
    std::vector<NnueAccumulator> accumulatorStack{accumulator};
    std::mt19937 random(0x4E4E5545U);
    for(int ply = 0; ply < 160; ply++){
        MoveList legal;
        board.genLegalMoves(legal);
        if(legal.empty()) break;
        const Move move = legal[static_cast<size_t>(random() % legal.size())];
        Undo undo{};
        if(!board.makeMove(move, undo)) continue;
        NnueAccumulator child;
        network.updateAfterMove(board, undo, accumulatorStack.back(), child);
        compare(board, child, "random playout ply " + std::to_string(ply));
        undos.push_back(undo);
        accumulatorStack.push_back(std::move(child));
    }
    while(!undos.empty()){
        board.undoMove(undos.back());
        undos.pop_back();
        accumulatorStack.pop_back();
        compare(board, accumulatorStack.back(), "random playout unmake");
    }

    PositionEvaluator evaluator;
    expect(evaluator.loadNnue(path, &error), "search NNUE fixture should load: " + error);
    expect(evaluator.setUseNnue(true), "search NNUE fixture should be selectable");
    Board incrementalBoard;
    incrementalBoard.setZobrist(&zobrist);
    incrementalBoard.reset();
    Board rebuildBoard = incrementalBoard;
    SearchContext incrementalSearch;
    incrementalSearch.evaluator = &evaluator;
    incrementalSearch.incrementalNnue = true;
    incrementalSearch.tt.resizeMB(16);
    incrementalSearch.gameHistory = {incrementalBoard.hash};
    SearchContext rebuildSearch;
    rebuildSearch.evaluator = &evaluator;
    rebuildSearch.incrementalNnue = false;
    rebuildSearch.tt.resizeMB(16);
    rebuildSearch.gameHistory = {rebuildBoard.hash};
    const Move incrementalMove = searchBestMove(incrementalBoard, incrementalSearch, 6, 60'000, 60'000);
    const Move rebuildMove = searchBestMove(rebuildBoard, rebuildSearch, 6, 60'000, 60'000);
    expect(sameMove(incrementalMove, rebuildMove),
           "incremental and rebuild NNUE searches should choose the same move");
    expect(incrementalSearch.stats.bestScore == rebuildSearch.stats.bestScore,
           "incremental and rebuild NNUE searches should return the same score");
    expect(incrementalSearch.stats.nodes == rebuildSearch.stats.nodes &&
           incrementalSearch.stats.qnodes == rebuildSearch.stats.qnodes,
           "incremental and rebuild NNUE searches should visit the same tree");

    std::error_code removeError;
    std::filesystem::remove(path, removeError);
}

void testStaticExchange(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    expect(board.loadFEN("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1"), "SEE fixture should load");
    const Move queenTakesPawn = findMove(board, "d1d5");
    expect(queenTakesPawn.from < 64, "SEE fixture capture should be legal");
    expect(staticExchangeEvaluation(board, queenTakesPawn) <= -700,
           "SEE should identify a queen taking a defended pawn as losing");

    expect(board.loadFEN("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"), "winning SEE fixture should load");
    const Move rookTakesQueen = findMove(board, "d2d5");
    expect(rookTakesQueen.from < 64, "winning SEE capture should be legal");
    expect(staticExchangeEvaluation(board, rookTakesQueen) == 900,
           "SEE should value an undefended queen capture");
}

void testContinuationHistoryOrdering(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    SearchContext context;
    const Move knightMove = findMove(board, "g1f3");
    expect(knightMove.from < 64, "continuation-history quiet move should be legal");

    Move previous = invalidMove();
    previous.from = static_cast<u8>(sqToIndex(Square{4, 6}));
    previous.to = static_cast<u8>(sqToIndex(Square{4, 4}));
    const int side = 0;
    context.continuationHistory[side][previous.to][knightMove.to] = 1234;
    expect(scoreMove(board, context, knightMove, invalidMove(), 0, previous) == 1234,
           "quiet ordering should include the previous-move continuation history");
}

void testStagedMovePicker(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    SearchContext context;
    MoveList moves;
    board.genLegalMoves(moves);
    const size_t legalCount = moves.size();

    const Move ttMove = findMove(board, "g1f3");
    const Move killer = findMove(board, "b1c3");
    const Move counter = findMove(board, "e2e4");
    context.killer[0][0] = killer;
    Move previous = invalidMove();
    previous.from = 1;
    previous.to = 18;
    context.countermove[0][previous.from][previous.to] = counter;

    StagedMovePicker picker(board, context, moves, ttMove, 0, previous);
    Move picked{};
    expect(picker.next(picked) && sameMove(picked, ttMove),
           "staged picker should emit the transposition move first");
    expect(picker.next(picked) && sameMove(picked, killer),
           "staged picker should emit the primary killer before ordinary quiets");
    expect(picker.next(picked) && sameMove(picked, counter),
           "staged picker should emit the countermove before ordinary quiets");

    std::vector<std::string> seen = {
        moveToUCI(ttMove), moveToUCI(killer), moveToUCI(counter)
    };
    while(picker.next(picked)){
        const std::string uci = moveToUCI(picked);
        expect(std::find(seen.begin(), seen.end(), uci) == seen.end(),
               "staged picker should never emit a move twice");
        seen.push_back(uci);
    }
    expect(seen.size() == legalCount,
           "staged picker should emit every legal move exactly once");

    expect(board.loadFEN("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1"),
           "winning staged-picker fixture should load");
    board.genLegalMoves(moves);
    StagedMovePicker winningPicker(board, context, moves, invalidMove(), 0, invalidMove());
    expect(winningPicker.next(picked) && moveToUCI(picked) == "d2d5",
           "winning captures should precede quiet moves");

    expect(board.loadFEN("4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1"),
           "losing staged-picker fixture should load");
    board.genLegalMoves(moves);
    StagedMovePicker losingPicker(board, context, moves, invalidMove(), 0, invalidMove());
    Move last{};
    while(losingPicker.next(picked)) last = picked;
    expect(moveToUCI(last) == "d1d5",
           "losing captures should follow ordinary quiet moves");
}

void testSearchPlySafety(const Zobrist& zobrist){
    static_assert(sizeof(StagedMovePicker) < 128,
                  "recursive move picker must not embed large stack arrays");
    SearchContext scratchContext;
    {
        SearchScratchScope parent(scratchContext);
        parent.frame().lists[0].push_back(Move{});
        auto* original = &parent.frame();
        {
            SearchScratchScope samePlyChild(scratchContext);
            expect(&samePlyChild.frame() != original, "same-ply recursion needs distinct scratch storage");
            expect(original->lists[0].size() == 1, "pool growth must preserve parent buffers");
        }
        expect(scratchContext.activeScratch == 1, "scratch scopes must release on return");
    }
    expect(scratchContext.activeScratch == 0, "all search scratch scopes must unwind");
    const auto allocated = scratchContext.scratch.size();
    { SearchScratchScope reuse(scratchContext);
      expect(reuse.frame().lists[0].empty(), "reused scratch lists must be cleared"); }
    expect(scratchContext.scratch.size() == allocated, "scratch storage must be reused");
    Board checkedRepetition;
    checkedRepetition.setZobrist(&zobrist);
    expect(checkedRepetition.loadFEN("4k3/8/8/8/8/8/4R3/4K3 b - - 10 1"),
           "checked repetition fixture should load");
    SearchContext repetitionContext;
    repetitionContext.start = std::chrono::steady_clock::now();
    repetitionContext.softTimeLimitMs = 60'000;
    repetitionContext.hardTimeLimitMs = 60'000;
    repetitionContext.repetition = {
        checkedRepetition.hash, checkedRepetition.hash, checkedRepetition.hash
    };
    expect(quiescence(checkedRepetition, repetitionContext, -INF, INF, 4) == 0,
           "quiescence should recognise a repeated position while in check");
    expect(repetitionContext.stats.qnodes == 1,
           "checked repetition should terminate without another quiescence ply");

    Board checkmate;
    checkmate.setZobrist(&zobrist);
    expect(checkmate.loadFEN("7k/6Q1/5K2/8/8/8/8/8 b - - 10 1"),
           "checkmate repetition fixture should load");
    SearchContext mateContext;
    mateContext.start = std::chrono::steady_clock::now();
    mateContext.softTimeLimitMs = 60'000;
    mateContext.hardTimeLimitMs = 60'000;
    mateContext.repetition = {checkmate.hash, checkmate.hash, checkmate.hash};
    expect(quiescence(checkmate, mateContext, -INF, INF, 4) == -MATE + 4,
           "checkmate should take precedence over a repetition draw");

    Board horizon;
    horizon.setZobrist(&zobrist);
    horizon.reset();
    SearchContext horizonContext;
    horizonContext.start = std::chrono::steady_clock::now();
    horizonContext.softTimeLimitMs = 60'000;
    horizonContext.hardTimeLimitMs = 60'000;
    horizonContext.repetition = {horizon.hash};
    const int expectedEvaluation = evaluateClassical(horizon);
    expect(quiescence(horizon, horizonContext, -INF, INF,
                      SearchContext::MaxSearchPly) == expectedEvaluation,
           "quiescence should return a bounded evaluation at the ply limit");

    SearchContext principalContext;
    principalContext.start = std::chrono::steady_clock::now();
    principalContext.softTimeLimitMs = 60'000;
    principalContext.hardTimeLimitMs = 60'000;
    principalContext.repetition = {horizon.hash};
    expect(negamax(horizon, principalContext, 4, -INF, INF,
                   SearchContext::MaxSearchPly, invalidMove(), true) == expectedEvaluation,
           "principal search should return a bounded evaluation at the ply limit");
}

void testSearchDeadlineAndBudget(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    expect(board.loadFEN("2N5/4kp2/1p4p1/4P3/p4P2/4P3/2K4p/4r3 b - - 3 58"),
           "timeout regression position should load");
    const Board original = board;
    auto ctx = std::make_unique<SearchContext>();
    ctx->tt.resizeMB(1);
    const auto expiredStart = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    const Move fallback = searchBestMove(board, *ctx, 64, 100, 100, 1, nullptr, expiredStart);
    MoveList legal;
    board.genLegalMoves(legal);
    expect(std::any_of(legal.begin(), legal.end(), [&](const Move& m){ return sameMove(m, fallback); }),
           "an expired UCI deadline must still return a legal fallback");
    expect(ctx->stats.nodes == 0 && ctx->stats.qnodes == 0 && ctx->stop,
           "worker delay must count against the deadline before searching any nodes");
    expect(samePosition(board, original), "expired search must preserve the board");

    for(int threads : {1, 2}){
        ctx->tt.resizeMB(1);
        searchBestMove(board, *ctx, 6, 60'000, 600'000, threads);
        expect(ctx->stats.depthReached == 6, "budget regression must finish all six iterations");
        // Even maximal instability cannot grow beyond these original-budget allowances.
        const int maximumSoft = 60'000 + 60'000 / 3 + 60'000 / 4 + 60'000 / 8;
        expect(ctx->stats.softTimeLimitMs <= maximumSoft,
               "iterative deepening must not compound earlier soft-budget extensions");
    }
}

void testClockTimeManagement(const Zobrist& zobrist){
    Board board;
    board.setZobrist(&zobrist);
    board.reset();

    const TimeBudget protectedClock = pickClockTimeBudget(board, 20, 20, -1, 25);
    expect(protectedClock.softMs == 1 && protectedClock.hardMs == 1,
           "move overhead should preserve a one-millisecond emergency clock budget");

    const TimeBudget smallerOverhead = pickClockTimeBudget(board, 20, 20, -1, 10);
    expect(smallerOverhead.softMs >= 1 && smallerOverhead.hardMs >= smallerOverhead.softMs,
           "low-clock budget should remain positive and ordered");
    expect(smallerOverhead.hardMs <= 10,
           "low-clock hard limit should not consume the configured reserve");

    const TimeBudget normalClock = pickClockTimeBudget(board, 60'000, 500, 30, 25);
    expect(normalClock.softMs > 0 && normalClock.hardMs >= normalClock.softMs,
           "normal clock budget should remain positive and ordered");
    expect(normalClock.hardMs <= 59'975,
           "normal clock hard limit should retain the configured move overhead");

    const TimeBudget manualGui = pickGuiTimeBudget(board, 10'000);
    expect(manualGui.softMs == 10'000 && manualGui.hardMs > manualGui.softMs,
           "manual GUI time should use the selected limit before a bounded overrun");

    const GuiTimeSuggestion eco = suggestGuiTimeLimit(board, 10'000, 300'000, GuiTimeProfile::Eco);
    const GuiTimeSuggestion balanced = suggestGuiTimeLimit(board, 10'000, 300'000, GuiTimeProfile::Balanced);
    const GuiTimeSuggestion performance = suggestGuiTimeLimit(board, 10'000, 300'000, GuiTimeProfile::Performance);
    const GuiTimeSuggestion performancePlus = suggestGuiTimeLimit(board, 10'000, 300'000, GuiTimeProfile::PerformancePlus);
    expect(eco.limitMs < balanced.limitMs && balanced.limitMs < performance.limitMs &&
               performance.limitMs < performancePlus.limitMs,
           "GUI automatic profiles should increase suggested thinking time monotonically");
    expect(eco.profileCapMs < balanced.profileCapMs && balanced.profileCapMs < performance.profileCapMs &&
               performance.profileCapMs < performancePlus.profileCapMs,
           "GUI automatic profiles should increase their effective cap monotonically");
    expect(performancePlus.profileCapMs == 300'000,
           "performance+ should be able to use the full five-minute automatic cap");
    expect(eco.profileCapMs == 30'000 && balanced.profileCapMs == 75'000 &&
               performance.profileCapMs == 150'000,
           "automatic profiles should use 10%, 25%, and 50% of the five-minute ceiling");
    expect(guiTimeProfileWeightPercent(GuiTimeProfile::Eco) == 55 &&
               guiTimeProfileWeightPercent(GuiTimeProfile::Balanced) == 85 &&
               guiTimeProfileWeightPercent(GuiTimeProfile::Performance) == 130 &&
               guiTimeProfileWeightPercent(GuiTimeProfile::PerformancePlus) == 180,
           "automatic profile complexity weights should remain progressively more aggressive");
}

void testParallelSearchSafety(const Zobrist& zobrist){
    Board singleBoard;
    singleBoard.setZobrist(&zobrist);
    singleBoard.reset();
    Board parallelBoard = singleBoard;

    SearchContext single;
    single.tt.resizeMB(16);
    single.gameHistory = {singleBoard.hash};
    const Move singleBest = searchBestMove(singleBoard, single, 2, 2000, 2000, 1);

    SearchContext parallel;
    parallel.tt.resizeMB(16);
    parallel.gameHistory = {parallelBoard.hash};
    const Move parallelBest = searchBestMove(parallelBoard, parallel, 2, 2000, 2000, 4);

    expect(sameMove(singleBest, parallelBest),
           "parallel root tie must preserve the fully searched principal move");
    expect(single.stats.bestScore == parallel.stats.bestScore,
           "parallel root tie must preserve the principal score");

    SearchContext shortSearch;
    shortSearch.tt.resizeMB(16);
    shortSearch.gameHistory = {parallelBoard.hash};
    const Move shortBest = searchBestMove(parallelBoard, shortSearch, 64, 10, 20, 4);
    expect(shortBest.from < 64, "short parallel request should still return a legal move");
    expect(shortSearch.stats.configuredThreads == 4,
           "short search should report the requested thread count");
    expect(shortSearch.stats.workersUsed == 1,
           "short search should avoid thread-startup overhead");
}

} // namespace

int main(){
    const Zobrist zobrist;
    const bool quick = std::getenv("CHESS_TEST_QUICK") != nullptr;
    testPerft(zobrist, quick);
    testMakeUnmakeAndHash(zobrist);
    testNullMoveAndHash(zobrist);
    testEvaluationOrientation(zobrist);
    testPseudoMobilityCounter(zobrist);
    testEnPassantHashing(zobrist);
    testDrawRules(zobrist);
    testNotation(zobrist);
    testFENValidation(zobrist);
    testTranspositionClusters();
    testConcurrentTranspositionTable();
    testNnueFormat(zobrist);
    testIncrementalNnue(zobrist);
    testStaticExchange(zobrist);
    testContinuationHistoryOrdering(zobrist);
    testStagedMovePicker(zobrist);
    testSearchPlySafety(zobrist);
    testSearchDeadlineAndBudget(zobrist);
    testClockTimeManagement(zobrist);
    testParallelSearchSafety(zobrist);

    if(failures != 0){
        std::cerr << failures << " test assertion(s) failed\n";
        return 1;
    }
    std::cout << "Core tests: PASS\n";
    return 0;
}
