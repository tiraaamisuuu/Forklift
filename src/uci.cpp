#include "uci.hpp"

#include "chess_core.hpp"
#include "time_management.hpp"

namespace {

bool startsWith(const std::string& value, const std::string& prefix){
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

std::string toLowerASCII(std::string value){
    for(char& ch : value){
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool parseIntStrict(const std::string& value, int& out){
    try{
        size_t parsedChars = 0;
        const int parsed = std::stoi(value, &parsedChars);
        if(parsedChars != value.size()) return false;
        out = parsed;
        return true;
    } catch(...){
        return false;
    }
}

bool parseUCIMove(Board& board, const std::string& uci, Move& out){
    std::vector<Move> legal;
    board.genLegalMoves(legal);
    const auto it = std::find_if(legal.begin(), legal.end(), [&](const Move& move){
        return moveToUCI(move) == uci;
    });
    if(it == legal.end()) return false;
    out = *it;
    return true;
}

bool applyPositionCommand(const std::string& line, Board& board, std::vector<u64>& history){
    std::istringstream input(line);
    std::string token;
    input >> token; // position

    std::vector<std::string> parts;
    while(input >> token) parts.push_back(token);
    if(parts.empty()) return false;

    size_t index = 0;
    if(parts[index] == "startpos"){
        board.reset();
        index++;
    } else if(parts[index] == "fen"){
        index++;
        const size_t fenStart = index;
        while(index < parts.size() && parts[index] != "moves") index++;
        if(index - fenStart != 6) return false;

        std::string fen;
        for(size_t i = fenStart; i < index; i++){
            if(!fen.empty()) fen.push_back(' ');
            fen += parts[i];
        }
        if(!board.loadFEN(fen)) return false;
    } else {
        return false;
    }

    history.assign(1, board.hash);
    if(index == parts.size()) return true;
    if(parts[index++] != "moves") return false;

    for(; index < parts.size(); index++){
        Move move{};
        if(!parseUCIMove(board, parts[index], move)) return false;
        Undo undo{};
        if(!board.makeMove(move, undo)) return false;
        history.push_back(board.hash);
    }
    return true;
}

struct GoParameters {
    int depth = -1;
    int movetimeMs = -1;
    int wtimeMs = -1;
    int btimeMs = -1;
    int wincMs = 0;
    int bincMs = 0;
    int movesToGo = -1;
    bool infinite = false;
    bool restrictRootMoves = false;
    std::vector<std::string> searchMoves;
};

GoParameters parseGoCommand(const std::string& line){
    GoParameters result;
    std::istringstream input(line);
    std::string token;
    input >> token; // go

    std::vector<std::string> tokens;
    while(input >> token) tokens.push_back(token);
    auto looksLikeMove = [](const std::string& value){
        if(value.size() != 4 && value.size() != 5) return false;
        if(value[0] < 'a' || value[0] > 'h' || value[2] < 'a' || value[2] > 'h' ||
           value[1] < '1' || value[1] > '8' || value[3] < '1' || value[3] > '8') return false;
        return value.size() == 4 || value[4] == 'q' || value[4] == 'r' ||
               value[4] == 'b' || value[4] == 'n';
    };

    for(size_t index = 0; index < tokens.size(); index++){
        auto readInt = [&](int& destination){
            if(index + 1 >= tokens.size()) return;
            int parsed = 0;
            if(parseIntStrict(tokens[++index], parsed)) destination = parsed;
        };

        const std::string& current = tokens[index];
        if(current == "depth") readInt(result.depth);
        else if(current == "movetime") readInt(result.movetimeMs);
        else if(current == "wtime") readInt(result.wtimeMs);
        else if(current == "btime") readInt(result.btimeMs);
        else if(current == "winc") readInt(result.wincMs);
        else if(current == "binc") readInt(result.bincMs);
        else if(current == "movestogo") readInt(result.movesToGo);
        else if(current == "infinite") result.infinite = true;
        else if(current == "searchmoves"){
            result.restrictRootMoves = true;
            while(index + 1 < tokens.size() && looksLikeMove(tokens[index + 1])){
                result.searchMoves.push_back(tokens[++index]);
            }
        }
    }
    return result;
}

TimeBudget pickUCITimeBudget(const Board& board, const GoParameters& parameters, int moveOverheadMs){
    if(parameters.movetimeMs > 0){
        return TimeBudget{parameters.movetimeMs, parameters.movetimeMs};
    }
    if(parameters.infinite){
        constexpr int oneDayMs = 24 * 60 * 60 * 1000;
        return TimeBudget{oneDayMs, oneDayMs};
    }

    const bool white = board.stm == Color::White;
    const int sideTime = white ? parameters.wtimeMs : parameters.btimeMs;
    const int sideIncrement = white ? parameters.wincMs : parameters.bincMs;
    if(sideTime < 0) return TimeBudget{1000, 1500};
    return pickClockTimeBudget(board, sideTime, sideIncrement,
                               parameters.movesToGo, moveOverheadMs);
}

} // namespace

int runUCILoop(int defaultThreads){
    Zobrist zobrist;
    Board board;
    board.setZobrist(&zobrist);
    board.reset();

    std::vector<u64> positionHistory{board.hash};
    const int hardwareThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    int searchThreads = std::clamp(defaultThreads, 1, hardwareThreads);
    int hashMB = 256;
    int moveOverheadMs = 25;

    PositionEvaluator evaluator;
    SearchContext search;
    search.tt.resizeMB(static_cast<size_t>(hashMB));
    search.evaluator = &evaluator;

    std::atomic<bool> abortSearch(false);
    std::mutex outputMutex;
    std::thread worker;

    auto stopSearch = [&](){
        abortSearch.store(true, std::memory_order_relaxed);
        if(worker.joinable()) worker.join();
        abortSearch.store(false, std::memory_order_relaxed);
    };

    auto resetSearch = [&](){
        stopSearch();
        search = SearchContext{};
        search.tt.resizeMB(static_cast<size_t>(hashMB));
        search.evaluator = &evaluator;
    };

    auto launchSearch = [&](int depth, TimeBudget budget, std::vector<Move> rootRestriction, bool restrictRootMoves,
                            std::chrono::steady_clock::time_point started){
        stopSearch();
        Board root = board;
        const std::vector<u64> rootHistory = positionHistory;
        const int threadsForSearch = searchThreads;
        abortSearch.store(false, std::memory_order_relaxed);

        worker = std::thread([&, root, rootHistory, depth, budget, threadsForSearch, started,
                              rootRestriction = std::move(rootRestriction), restrictRootMoves]() mutable {
            search.abortFlag = &abortSearch;
            search.gameHistory = rootHistory;
            search.evaluator = &evaluator;
            const std::vector<Move>* restriction = restrictRootMoves ? &rootRestriction : nullptr;
            Move best = searchBestMove(root, search, depth, budget.softMs, budget.hardMs,
                                       threadsForSearch, restriction, started);

            std::vector<Move> legal;
            root.genLegalMoves(legal);
            if(restrictRootMoves) legal = rootRestriction;
            const auto legalBest = std::find_if(legal.begin(), legal.end(), [&](const Move& move){
                return sameMove(move, best);
            });
            if(legalBest == legal.end() && !legal.empty()) best = legal.front();

            const u64 nodes = search.stats.nodes + search.stats.qnodes;
            const long long nps = search.stats.timeMs > 0
                ? static_cast<long long>((nodes * 1000ULL) / static_cast<u64>(search.stats.timeMs))
                : 0LL;
            const std::string pv = extractPVFromTT(root, search, 16);

            std::lock_guard<std::mutex> lock(outputMutex);
            std::cout << "info depth " << search.stats.depthReached;
            if(std::abs(search.stats.bestScore) >= MATE - 10000){
                const int plies = MATE - std::abs(search.stats.bestScore);
                const int moves = std::max(1, (plies + 1) / 2);
                std::cout << " score mate " << (search.stats.bestScore >= 0 ? moves : -moves);
            } else {
                std::cout << " score cp " << search.stats.bestScore;
            }
            std::cout << " nodes " << nodes
                      << " nps " << nps
                      << " time " << search.stats.timeMs;
            if(!pv.empty()) std::cout << " pv " << pv;
            std::cout << "\nbestmove " << (legal.empty() ? "0000" : moveToUCI(best)) << "\n" << std::flush;
        });
    };

    std::string line;
    while(std::getline(std::cin, line)){
        line = trim(line);
        if(line.empty()) continue;
        const std::string lower = toLowerASCII(line);

        if(lower == "uci"){
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cout << "id name Forklift v1.1.0\n"
                      << "id author Alfie Corthine\n"
                      << "option name Hash type spin default 256 min 1 max 4096\n"
                      << "option name Threads type spin default " << searchThreads
                      << " min 1 max " << hardwareThreads << "\n"
                      << "option name Move Overhead type spin default 25 min 0 max 5000\n"
                      << "option name Clear Hash type button\n"
                      << "option name EvalFile type string default <empty>\n"
                      << "option name Use NNUE type check default false\n"
                      << "option name NNUE Weight type spin default 100 min 0 max 100\n"
                      << "uciok\n" << std::flush;
        } else if(lower == "isready"){
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cout << "readyok\n" << std::flush;
        } else if(startsWith(lower, "setoption")){
            const size_t valuePosition = lower.find(" value ");
            if(lower.find("name clear hash") != std::string::npos){
                resetSearch();
            } else if(lower.find("name evalfile") != std::string::npos && valuePosition != std::string::npos){
                stopSearch();
                const std::string path = trim(line.substr(valuePosition + 7));
                std::string error;
                const bool loaded = !path.empty() && path != "<empty>" && evaluator.loadNnue(path, &error);
                resetSearch();
                std::lock_guard<std::mutex> lock(outputMutex);
                if(loaded){
                    std::cout << "info string NNUE loaded: " << path << "\n" << std::flush;
                } else {
                    std::cout << "info string NNUE load failed: "
                              << (error.empty() ? "no network path" : error) << "\n" << std::flush;
                }
            } else if(lower.find("name use nnue") != std::string::npos && valuePosition != std::string::npos){
                stopSearch();
                const std::string value = toLowerASCII(trim(line.substr(valuePosition + 7)));
                const bool enable = value == "true" || value == "1" || value == "on";
                const bool accepted = evaluator.setUseNnue(enable);
                resetSearch();
                if(!accepted){
                    std::lock_guard<std::mutex> lock(outputMutex);
                    std::cout << "info string Use NNUE requires a valid EvalFile\n" << std::flush;
                }
            } else if(lower.find("name nnue weight") != std::string::npos && valuePosition != std::string::npos){
                int value = 100;
                if(parseIntStrict(trim(line.substr(valuePosition + 7)), value)){
                    stopSearch();
                    evaluator.setNnueWeight(value);
                    resetSearch();
                }
            } else if(lower.find("name hash") != std::string::npos && valuePosition != std::string::npos){
                int value = 0;
                if(parseIntStrict(trim(line.substr(valuePosition + 7)), value)){
                    hashMB = std::clamp(value, 1, 4096);
                    resetSearch();
                }
            } else if(lower.find("name threads") != std::string::npos && valuePosition != std::string::npos){
                int value = 1;
                if(parseIntStrict(trim(line.substr(valuePosition + 7)), value)){
                    searchThreads = std::clamp(value, 1, hardwareThreads);
                }
            } else if(lower.find("name move overhead") != std::string::npos && valuePosition != std::string::npos){
                int value = 25;
                if(parseIntStrict(trim(line.substr(valuePosition + 7)), value)){
                    moveOverheadMs = std::clamp(value, 0, 5000);
                }
            }
        } else if(lower == "ucinewgame"){
            board.reset();
            positionHistory.assign(1, board.hash);
            resetSearch();
        } else if(startsWith(lower, "position ")){
            stopSearch();
            if(!applyPositionCommand(line, board, positionHistory)){
                std::lock_guard<std::mutex> lock(outputMutex);
                std::cout << "info string invalid position command\n" << std::flush;
            }
        } else if(startsWith(lower, "go")){
            const auto started = std::chrono::steady_clock::now();
            const GoParameters parameters = parseGoCommand(line);
            constexpr int oneDayMs = 24 * 60 * 60 * 1000;
            TimeBudget budget = pickUCITimeBudget(board, parameters, moveOverheadMs);
            budget.softMs = std::clamp(budget.softMs, 1, oneDayMs);
            budget.hardMs = std::clamp(budget.hardMs, budget.softMs, oneDayMs);
            std::vector<Move> rootRestriction;
            for(const std::string& uciMove : parameters.searchMoves){
                Move move{};
                if(parseUCIMove(board, uciMove, move) &&
                   std::none_of(rootRestriction.begin(), rootRestriction.end(), [&](const Move& existing){
                       return sameMove(existing, move);
                   })){
                    rootRestriction.push_back(move);
                }
            }
            launchSearch(parameters.depth > 0 ? parameters.depth : 64, budget,
                         std::move(rootRestriction), parameters.restrictRootMoves, started);
        } else if(lower == "stop"){
            stopSearch();
        } else if(lower == "quit"){
            stopSearch();
            return 0;
        } else if(lower == "ponderhit" || lower == "debug on" || lower == "debug off" || lower == "d"){
            continue;
        } else {
            std::lock_guard<std::mutex> lock(outputMutex);
            std::cout << "info string unknown command: " << line << "\n" << std::flush;
        }
    }

    stopSearch();
    return 0;
}
