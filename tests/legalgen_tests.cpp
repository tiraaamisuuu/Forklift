#include "chess_core.hpp"

#include <random>

namespace {

int failures = 0;
u64 comparedPositions = 0;
u64 comparedMoves = 0;

void expect(bool condition, const std::string& message){
    if(condition) return;
    std::cerr << "[FAIL] " << message << '\n';
    failures++;
}

bool sameState(const Board& left, const Board& right){
    if(left.stm != right.stm || left.epSquare != right.epSquare ||
       left.castling != right.castling || left.halfmoveClock != right.halfmoveClock ||
       left.fullmoveNumber != right.fullmoveNumber || left.kingSquare != right.kingSquare ||
       left.hash != right.hash || left.z != right.z) return false;
    for(size_t square = 0; square < left.b.size(); square++){
        if(left.b[square].t != right.b[square].t || left.b[square].c != right.b[square].c) return false;
    }
    return true;
}

bool sameMetadata(const Move& left, const Move& right){
    return left.from == right.from && left.to == right.to && left.promo == right.promo &&
           left.isCapture == right.isCapture && left.isEnPassant == right.isEnPassant &&
           left.isCastle == right.isCastle;
}

// Deliberately independent of both public generators: preserve the original
// generate-pseudo / make-and-undo filter as an ordered differential oracle.
std::vector<Move> independentReference(Board& board){
    std::vector<Move> pseudo;
    board.genPseudoMoves(pseudo);
    std::vector<Move> legal;
    for(const Move& move : pseudo){
        Undo undo{};
        if(board.makeMove(move, undo)){
            legal.push_back(move);
            board.undoMove(undo);
        }
    }
    return legal;
}

template<typename Container>
void compareOrdered(const Container& actual, const std::vector<Move>& expected,
                    const std::string& label){
    expect(actual.size() == expected.size(), label + " ordered move count");
    for(size_t index = 0; index < std::min(actual.size(), expected.size()); index++){
        expect(sameMetadata(actual[index], expected[index]),
               label + " exact ordered metadata at " + std::to_string(index));
    }
}

std::vector<Move> compareGenerators(Board& board, const std::string& label){
    const Board original = board;
    const std::vector<Move> oracle = independentReference(board);
    expect(sameState(board, original), label + " independent oracle preserves all state");
    std::vector<Move> fast;
    board.genLegalMoves(fast);
    expect(sameState(board, original), label + " vector fast generator preserves all state");
    compareOrdered(fast, oracle, label + " vector fast");
    std::vector<Move> reference;
    board.genLegalMovesReference(reference);
    expect(sameState(board, original), label + " public reference preserves all state");
    compareOrdered(reference, oracle, label + " public reference");
    MoveList fixed;
    board.genLegalMoves(fixed);
    expect(sameState(board, original), label + " fixed fast generator preserves all state");
    compareOrdered(fixed, oracle, label + " fixed fast");
    comparedPositions++;
    comparedMoves += static_cast<u64>(oracle.size());
    return oracle;
}

bool containsUci(const std::vector<Move>& moves, const std::string& uci){
    return std::any_of(moves.begin(), moves.end(), [&](const Move& move){
        return moveToUCI(move) == uci;
    });
}

void testFixtures(const Zobrist& zobrist){
    struct Fixture { const char* name; const char* fen; const char* forbidden; const char* required; };
    const Fixture fixtures[] = {
        {"start", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "", "e2e4"},
        {"both white castles", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 19 24", "", "e1g1"},
        {"both black castles", "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 19 24", "", "e8c8"},
        {"castle through attack", "r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1", "e1g1", "e1c1"},
        {"black castle through attack", "r3k2r/8/5R2/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", "e8c8"},
        {"castle absent rook", "4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1", "e1g1", ""},
        {"legal en passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "", "e5d6"},
        {"en passant exposes horizontal rook", "7k/8/8/r4pPK/8/8/8/8 w - f6 0 1", "g5f6", ""},
        {"en passant exposes vertical rook", "4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", ""},
        {"en passant removes bishop blocker", "7k/8/5b2/3Pp3/8/2K5/8/8 w - e6 0 1", "d5e6", ""},
        {"en passant captures checking pawn", "7k/8/8/3pP3/4K3/8/8/8 w - d6 0 1", "", "e5d6"},
        {"black legal en passant", "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "", "e4d3"},
        {"black en passant exposes horizontal rook", "8/8/8/8/R4Ppk/8/8/7K b - f3 0 1", "g4f3", ""},
        {"black en passant exposes vertical rook", "4k3/8/8/8/3Pp3/8/8/4R2K b - d3 0 1", "e4d3", ""},
        {"black en passant captures checking pawn", "7K/8/8/4k3/3Pp3/8/8/8 b - d3 0 1", "", "e4d3"},
        {"white quiet and capture promotions", "1r5k/P7/8/8/8/8/8/4K3 w - - 0 1", "", "a7b8n"},
        {"black quiet and capture promotions", "4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1", "", "a2b1n"},
        {"double check", "4r2k/8/8/8/1b6/8/8/4K3 w - - 0 1", "", "e1f1"},
        {"single-check blocking", "4r2k/8/8/8/8/8/R7/4K3 w - - 0 1", "a2a3", "a2e2"},
        {"checkmate", "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", "", ""},
        {"stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "", ""},
        // loadFEN accepts some non-legal chess arrangements. This optimisation
        // must preserve legacy makeMove behaviour, not silently repair rules.
        {"legacy king-target capture", "4k3/4R3/8/8/8/8/8/4K3 w - - 0 1", "", "e7e8"},
    };
    for(const Fixture& fixture : fixtures){
        Board board;
        board.setZobrist(&zobrist);
        expect(board.loadFEN(fixture.fen), std::string(fixture.name) + " loads");
        const auto moves = compareGenerators(board, fixture.name);
        if(fixture.forbidden[0]) expect(!containsUci(moves, fixture.forbidden),
                                       std::string(fixture.name) + " forbids " + fixture.forbidden);
        if(fixture.required[0]) expect(containsUci(moves, fixture.required),
                                      std::string(fixture.name) + " requires " + fixture.required);
        if(std::string(fixture.name) == "double check"){
            for(const Move& move : moves) expect(board.b[move.from].t == PieceType::King,
                                                "double check only permits king moves");
        }
        if(std::string(fixture.name).find("promotions") != std::string::npos){
            expect(std::count_if(moves.begin(), moves.end(), [](const Move& move){
                return move.promo != PieceType::None;
            }) == 8, std::string(fixture.name) + " includes all eight under/queen promotions");
        }
    }
}

void testAllPinRays(const Zobrist& zobrist){
    const int directions[8][2] = {{1,0},{-1,0},{0,1},{0,-1},{1,1},{1,-1},{-1,1},{-1,-1}};
    for(int direction = 0; direction < 8; direction++){
        const int df = directions[direction][0];
        const int dr = directions[direction][1];
        Board board;
        board.clear();
        board.setZobrist(&zobrist);
        board.castling = 0;
        board.b[27] = Piece{PieceType::King, Color::White};  // d4
        board.b[56] = Piece{PieceType::King, Color::Black};  // a8
        board.kingSquare = {{27, 56}};
        const int pinned = 27 + df + dr * 8;
        const int attacker = 27 + 2 * df + 2 * dr * 8;
        board.b[static_cast<size_t>(pinned)] = Piece{PieceType::Queen, Color::White};
        board.b[static_cast<size_t>(attacker)] = Piece{
            (df == 0 || dr == 0) ? PieceType::Rook : PieceType::Bishop, Color::Black};
        board.b[15] = Piece{PieceType::Knight, Color::White};
        board.recomputeHash();
        expect(!board.inCheck(Color::White), "pinned queen initially shields king");
        compareGenerators(board, "pin direction " + std::to_string(direction));

        // A second friendly blocker means the nearer piece is not absolutely
        // pinned. This catches false positives in ray classification as well.
        board.b[static_cast<size_t>(attacker)].t = PieceType::Pawn;
        board.b[static_cast<size_t>(27 + 3 * df + 3 * dr * 8)] = Piece{
            (df == 0 || dr == 0) ? PieceType::Rook : PieceType::Bishop, Color::Black};
        board.b[static_cast<size_t>(attacker)].c = Color::White;
        board.recomputeHash();
        compareGenerators(board, "two blockers direction " + std::to_string(direction));
    }
}

void testMalformedFallbacks(const Zobrist& zobrist){
    const char* malformed[] = {
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "4k3/8/8/8/8/8/8/8 w - - 0 1",
        "8/8/8/8/8/8/8/4K3 b - - 0 1",
        "4k3/8/8/8/8/8/4K3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w Z - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - z9 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - -",
    };
    for(size_t index = 0; index < std::size(malformed); index++){
        Board board;
        board.setZobrist(&zobrist);
        expect(!board.loadFEN(malformed[index]), "malformed FEN is rejected");
        compareGenerators(board, "state after rejected FEN " + std::to_string(index));
    }
    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    board.kingSquare[0] = -1;
    compareGenerators(board, "missing cached own king");
    board.reset();
    board.kingSquare[1] = 0;
    compareGenerators(board, "inconsistent opponent king cache");
    board.reset();
    board.kingSquare[0] = 0;
    compareGenerators(board, "inconsistent own king cache");
    board.reset();
    board.stm = static_cast<Color>(3);
    compareGenerators(board, "invalid side enumeration");
    board.setZobrist(nullptr);
    board.reset();
    compareGenerators(board, "no Zobrist table");
}

void testSeededArtificialPositions(const Zobrist& zobrist){
    // Structurally consistent boards need not be reachable legal positions.
    // Their impossible checks, promoted material and king-target captures are
    // useful compatibility inputs: this optimisation must preserve the old
    // filter rather than silently change its accepted rules or move order.
    std::mt19937 random(0x41525449U);
    for(int sample = 0; sample < 2048; sample++){
        Board board;
        board.clear();
        board.setZobrist(&zobrist);
        board.castling = 0;
        board.stm = sample % 2 == 0 ? Color::White : Color::Black;
        const int whiteKing = static_cast<int>(random() % 64);
        int blackKing = static_cast<int>(random() % 63);
        if(blackKing >= whiteKing) blackKing++;
        board.b[static_cast<size_t>(whiteKing)] = Piece{PieceType::King, Color::White};
        board.b[static_cast<size_t>(blackKing)] = Piece{PieceType::King, Color::Black};
        board.kingSquare = {{whiteKing, blackKing}};
        const int additionalPieces = static_cast<int>(random() % 15);
        for(int piece = 0; piece < additionalPieces; piece++){
            int square = static_cast<int>(random() % 64);
            while(!isNone(board.b[static_cast<size_t>(square)])){
                square = (square + 1) % 64;
            }
            PieceType type = static_cast<PieceType>(1 + random() % 5);
            // Keep pawn locations compatible with ordinary chess structure;
            // promotions themselves are exercised by the special fixtures.
            if(type == PieceType::Pawn && (square / 8 == 0 || square / 8 == 7)){
                type = PieceType::Knight;
            }
            board.b[static_cast<size_t>(square)] = Piece{
                type, random() % 2 == 0 ? Color::White : Color::Black};
        }
        board.recomputeHash();
        compareGenerators(board, "artificial position " + std::to_string(sample));
    }
}

void testSeededPlayouts(const Zobrist& zobrist){
    std::mt19937 random(0x4C454741U);
    const u64 before = comparedPositions;
    for(int game = 0; game < 32; game++){
        Board board;
        board.setZobrist(&zobrist);
        board.reset();
        const Board initial = board;
        std::vector<Undo> history;
        for(int ply = 0; ply < 128; ply++){
            const auto moves = compareGenerators(board, "playout " + std::to_string(game) +
                                                        " ply " + std::to_string(ply));
            if(moves.empty()) break;
            Undo undo{};
            expect(board.makeMove(moves[static_cast<size_t>(random() % moves.size())], undo),
                   "oracle move makes successfully during legal playout");
            history.push_back(undo);
            const u64 incremental = board.hash;
            board.recomputeHash();
            expect(board.hash == incremental, "playout hash agrees with recomputation");
        }
        while(!history.empty()){
            board.undoMove(history.back());
            history.pop_back();
        }
        expect(sameState(board, initial), "complete seeded playout unwinds exactly");
    }
    expect(comparedPositions - before >= 3000, "at least three thousand legal playout positions compared");
}

u64 referencePerft(Board& board, int depth){
    if(depth == 0) return 1;
    const auto moves = independentReference(board);
    u64 count = 0;
    for(const Move& move : moves){
        Undo undo{};
        if(!board.makeMove(move, undo)) continue;
        count += referencePerft(board, depth - 1);
        board.undoMove(undo);
    }
    return count;
}

void testStandardPerfts(const Zobrist& zobrist){
    struct Fixture { const char* fen; int depth; u64 count; };
    const Fixture fixtures[] = {
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281},
        {"r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQBBPPP/R3K2R w KQkq - 0 1", 3, 85877},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3, 2812},
        {"r3k2r/Pppp1ppp/1b3nbN/nP6/B1P1P3/5N2/Pp1P1PPP/R2Q1RK1 w kq - 0 1", 3, 35941},
        {"rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379},
        {"r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP1QPPP/R4RK1 w - - 0 10", 3, 83034},
    };
    for(const Fixture& fixture : fixtures){
        Board board;
        board.setZobrist(&zobrist);
        expect(board.loadFEN(fixture.fen), "standard perft fixture loads");
        const Board original = board;
        expect(perft(board, fixture.depth) == fixture.count, "fast standard perft matches known count");
        expect(sameState(board, original), "fast perft preserves all state");
        expect(referencePerft(board, fixture.depth) == fixture.count,
               "independent reference perft matches known count");
        expect(sameState(board, original), "independent reference perft preserves all state");
    }
}

} // namespace

int main(){
    const Zobrist zobrist;
    testFixtures(zobrist);
    testAllPinRays(zobrist);
    testMalformedFallbacks(zobrist);
    testSeededArtificialPositions(zobrist);
    testSeededPlayouts(zobrist);
    testStandardPerfts(zobrist);
    std::cout << "legalgen positions=" << comparedPositions << " orderedMoves=" << comparedMoves
              << " failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
