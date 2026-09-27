#include "chess_core.hpp"

#include <iostream>
#include <string>

namespace {

bool parseInteger(const std::string& value, int& out){
    try{
        size_t parsedCharacters = 0;
        const int parsed = std::stoi(value, &parsedCharacters);
        if(parsedCharacters != value.size()) return false;
        out = parsed;
        return true;
    } catch(...){
        return false;
    }
}

void printHelp(){
    std::cout
        << "Usage:\n"
        << "  chess-engine-tools --perft DEPTH [--fen FEN]\n"
        << "  chess-engine-tools --divide DEPTH [--fen FEN]\n"
        << "  chess-engine-tools --perft-tests [--max-depth DEPTH]\n"
        << "  chess-engine-tools --fen-after \"UCI MOVES...\" [--fen FEN]\n"
        << "  chess-engine-tools --eval [--fen FEN] [--nnue NETWORK] [--nnue-weight PERCENT]\n"
        << "  chess-engine-tools --bench [--bench-depth DEPTH] [--bench-time MS]\n"
        << "                  [--bench-tt MB] [--threads N] [--nnue NETWORK]\n"
        << "                  [--nnue-weight PERCENT]\n"
        << "                  [--nnue-rebuild] [--no-probcut]\n";
}

} // namespace

int main(int argc, char** argv){
    int perftDepth = -1;
    bool divide = false;
    bool perftTests = false;
    bool benchmark = false;
    bool evaluate = false;
    bool nnueRebuild = false;
    bool enableProbCut = true;
    int maxPerftDepth = 4;
    int benchmarkDepth = 8;
    int benchmarkTimeMs = 4000;
    int benchmarkTTMB = 256;
    int threads = 1;
    int nnueWeight = 100;
    std::string fen;
    std::string movesForFen;
    std::string nnuePath;

    for(int index = 1; index < argc; index++){
        const std::string argument = argv[index];
        auto valueFor = [&](const char* option)->const char*{
            if(index + 1 < argc) return argv[++index];
            std::cerr << "Missing value for " << option << '\n';
            return nullptr;
        };

        if(argument == "--help" || argument == "-h"){
            printHelp();
            return 0;
        }
        if(argument == "--perft" || argument == "--divide"){
            const char* value = valueFor(argument.c_str());
            if(!value || !parseInteger(value, perftDepth) || perftDepth < 0) return 1;
            divide = argument == "--divide";
        } else if(argument == "--fen"){
            const char* value = valueFor("--fen");
            if(!value) return 1;
            fen = value;
        } else if(argument == "--fen-after"){
            const char* value = valueFor("--fen-after");
            if(!value) return 1;
            movesForFen = value;
        } else if(argument == "--perft-tests"){
            perftTests = true;
        } else if(argument == "--eval"){
            evaluate = true;
        } else if(argument == "--nnue"){
            const char* value = valueFor("--nnue");
            if(!value) return 1;
            nnuePath = value;
        } else if(argument == "--nnue-weight"){
            const char* value = valueFor("--nnue-weight");
            if(!value || !parseInteger(value, nnueWeight) || nnueWeight < 0 || nnueWeight > 100){
                std::cerr << "--nnue-weight must be between 0 and 100\n";
                return 1;
            }
        } else if(argument == "--nnue-rebuild"){
            nnueRebuild = true;
        } else if(argument == "--no-probcut"){
            enableProbCut = false;
        } else if(argument == "--max-depth"){
            const char* value = valueFor("--max-depth");
            if(!value || !parseInteger(value, maxPerftDepth) || maxPerftDepth < 1) return 1;
        } else if(argument == "--bench"){
            benchmark = true;
        } else if(argument == "--bench-depth"){
            const char* value = valueFor("--bench-depth");
            if(!value || !parseInteger(value, benchmarkDepth) || benchmarkDepth < 1) return 1;
        } else if(argument == "--bench-time"){
            const char* value = valueFor("--bench-time");
            if(!value || !parseInteger(value, benchmarkTimeMs) || benchmarkTimeMs < 50) return 1;
        } else if(argument == "--bench-tt"){
            const char* value = valueFor("--bench-tt");
            if(!value || !parseInteger(value, benchmarkTTMB) || benchmarkTTMB < 1) return 1;
        } else if(argument == "--threads"){
            const char* value = valueFor("--threads");
            if(!value || !parseInteger(value, threads) || threads < 1 || threads > 64) return 1;
        } else {
            std::cerr << "Unknown argument: " << argument << '\n';
            printHelp();
            return 1;
        }
    }

    const Zobrist zobrist;
    if(perftTests) return runPerftSuite(zobrist, maxPerftDepth);

    PositionEvaluator evaluator;
    if(nnueRebuild && (!benchmark || nnuePath.empty())){
        std::cerr << "--nnue-rebuild requires --bench and --nnue\n";
        return 1;
    }
    if(nnueWeight != 100 && nnuePath.empty()){
        std::cerr << "--nnue-weight requires --nnue\n";
        return 1;
    }
    if(!nnuePath.empty()){
        std::string error;
        if(!evaluator.loadNnue(nnuePath, &error)){
            std::cerr << "Unable to load NNUE network: " << error << '\n';
            return 1;
        }
        evaluator.setUseNnue(true);
        evaluator.setNnueWeight(nnueWeight);
    }
    if(benchmark){
        return runSearchBenchmark(
            zobrist, benchmarkDepth, benchmarkTimeMs, benchmarkTTMB, threads,
            nnuePath.empty() ? nullptr : &evaluator, !nnueRebuild, enableProbCut);
    }

    if(evaluate){
        Board board;
        board.setZobrist(&zobrist);
        board.reset();
        if(!fen.empty() && !board.loadFEN(fen)){
            std::cerr << "Invalid FEN\n";
            return 1;
        }
        std::cout << "evaluation_cp=" << evaluator.evaluate(board)
                  << " backend=" << evaluator.backendName()
                  << " nnue_weight=" << evaluator.nnueWeight() << '\n';
        return 0;
    }

    if(!movesForFen.empty()){
        Board board;
        board.setZobrist(&zobrist);
        board.reset();
        if(!fen.empty() && !board.loadFEN(fen)){
            std::cerr << "Invalid initial FEN\n";
            return 1;
        }

        std::istringstream input(movesForFen);
        std::string uci;
        while(input >> uci){
            std::vector<Move> legal;
            board.genLegalMoves(legal);
            const auto move = std::find_if(legal.begin(), legal.end(), [&](const Move& candidate){
                return moveToUCI(candidate) == uci;
            });
            if(move == legal.end()){
                std::cerr << "Illegal move in line: " << uci << '\n';
                return 1;
            }
            Undo undo{};
            board.makeMove(*move, undo);
        }
        std::cout << board.toFEN() << '\n';
        return 0;
    }

    if(perftDepth >= 0){
        Board board;
        board.setZobrist(&zobrist);
        board.reset();
        if(!fen.empty() && !board.loadFEN(fen)){
            std::cerr << "Invalid FEN\n";
            return 1;
        }

        const auto started = std::chrono::steady_clock::now();
        if(divide){
            const auto results = perftDivide(board, perftDepth);
            u64 total = 0;
            for(const auto& [move, nodes] : results){
                std::cout << move << ": " << nodes << '\n';
                total += nodes;
            }
            std::cout << "Total: " << total << '\n';
        } else {
            const u64 nodes = perft(board, perftDepth);
            const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - started).count();
            std::cout << "Perft(" << perftDepth << ") = " << nodes
                      << " nodes in " << elapsed << " ms\n";
        }
        return 0;
    }

    printHelp();
    return 1;
}
