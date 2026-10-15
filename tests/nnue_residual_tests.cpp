#include "nnue.hpp"

#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const std::string& message){
    if(!condition) throw std::runtime_error(message);
}

struct Fixture {
    u32 hidden = 1;
    int32_t activationScale = 16;
    int32_t outputScale = 3;
    int32_t outputBias = 11;
    u32 residualHidden = 1;
    int32_t residualWeightScale = 7;
    std::vector<int32_t> hiddenBias;
    std::vector<int16_t> input;
    std::vector<int16_t> direct;
    std::vector<int32_t> residualBias;
    std::vector<int16_t> residualWeights;
    std::vector<int16_t> residualOutput;
};

Fixture patterned(u32 hidden, u32 residual){
    Fixture fixture;
    fixture.hidden = hidden;
    fixture.residualHidden = residual;
    fixture.hiddenBias.resize(hidden);
    fixture.input.resize(static_cast<size_t>(NnueNetwork::FeatureCount) * hidden);
    fixture.direct.resize(static_cast<size_t>(hidden) * 2);
    fixture.residualBias.resize(residual);
    fixture.residualWeights.resize(static_cast<size_t>(residual) * hidden * 2);
    fixture.residualOutput.resize(residual);
    for(size_t unit = 0; unit < hidden; unit++) fixture.hiddenBias[unit] = static_cast<int32_t>(unit % 13) - 4;
    for(size_t index = 0; index < fixture.input.size(); index++){
        fixture.input[index] = static_cast<int16_t>(static_cast<int>((index * 13 + index / hidden) % 17) - 8);
    }
    for(size_t index = 0; index < fixture.direct.size(); index++){
        fixture.direct[index] = static_cast<int16_t>(static_cast<int>((index * 7) % 31) - 15);
    }
    for(size_t row = 0; row < residual; row++){
        fixture.residualBias[row] = static_cast<int32_t>(row * 19) - 27;
        fixture.residualOutput[row] = static_cast<int16_t>(static_cast<int>(row * 11 % 37) - 18);
    }
    for(size_t index = 0; index < fixture.residualWeights.size(); index++){
        fixture.residualWeights[index] = static_cast<int16_t>(static_cast<int>((index * 5 + index / (2 * hidden)) % 23) - 11);
    }
    return fixture;
}

void append16(std::vector<unsigned char>& bytes, int16_t value){
    const auto bits = static_cast<uint16_t>(value);
    bytes.push_back(static_cast<unsigned char>(bits & 255U));
    bytes.push_back(static_cast<unsigned char>(bits >> 8));
}

void append32(std::vector<unsigned char>& bytes, u32 value){
    for(unsigned shift = 0; shift < 32; shift += 8){
        bytes.push_back(static_cast<unsigned char>((value >> shift) & 255U));
    }
}

void replace32(std::vector<unsigned char>& bytes, size_t offset, u32 value){
    require(offset + 4 <= bytes.size(), "fixture patch outside header");
    for(unsigned index = 0; index < 4; index++){
        bytes[offset + index] = static_cast<unsigned char>((value >> (8 * index)) & 255U);
    }
}

std::vector<unsigned char> encode(const Fixture& fixture, bool legacy = false){
    std::vector<unsigned char> bytes{'T', 'N', 'N', 'U', 'E',
        static_cast<unsigned char>(legacy ? '1' : '2'), 0, 0};
    append32(bytes, legacy ? 1U : 2U);
    append32(bytes, NnueNetwork::FeatureCount);
    append32(bytes, fixture.hidden);
    append32(bytes, static_cast<u32>(fixture.activationScale));
    append32(bytes, static_cast<u32>(fixture.outputScale));
    append32(bytes, static_cast<u32>(fixture.outputBias));
    if(!legacy){
        append32(bytes, fixture.residualHidden);
        append32(bytes, static_cast<u32>(fixture.residualWeightScale));
    }
    for(const int32_t value : fixture.hiddenBias) append32(bytes, static_cast<u32>(value));
    for(const int16_t value : fixture.input) append16(bytes, value);
    for(const int16_t value : fixture.direct) append16(bytes, value);
    if(!legacy){
        for(const int32_t value : fixture.residualBias) append32(bytes, static_cast<u32>(value));
        for(const int16_t value : fixture.residualWeights) append16(bytes, value);
        for(const int16_t value : fixture.residualOutput) append16(bytes, value);
    }
    return bytes;
}

// Every file is owned by this test. Rewrites are confined to the unique path.
struct FixtureFile {
    std::filesystem::path path;
    FixtureFile(){
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("forklift-nnue-residual-test-" +
            std::to_string(stamp) + "-" + std::to_string(std::random_device{}()) + ".nnue");
        require(!std::filesystem::exists(path), "temporary filename collision");
    }
    ~FixtureFile(){
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    void write(const std::vector<unsigned char>& bytes) const {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if(!bytes.empty()){
            output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
        require(static_cast<bool>(output), "could not write residual test fixture");
    }
};

// Deliberately independent of the production output helpers and vector layout.
int reference(const Fixture& fixture, const NnueAccumulator& accumulator, size_t stm,
              bool legacy = false){
    std::vector<int64_t> activation(static_cast<size_t>(fixture.hidden) * 2);
    for(size_t half = 0; half < 2; half++){
        for(size_t unit = 0; unit < fixture.hidden; unit++){
            activation[half * fixture.hidden + unit] = std::clamp<int64_t>(
                accumulator.values[half == 0 ? stm : 1 - stm][unit], 0, fixture.activationScale);
        }
    }
    int64_t output = fixture.outputBias;
    for(size_t unit = 0; unit < activation.size(); unit++) output += activation[unit] * fixture.direct[unit];
    if(!legacy){
        for(size_t row = 0; row < fixture.residualHidden; row++){
            int64_t preactivation = fixture.residualBias[row];
            for(size_t unit = 0; unit < activation.size(); unit++){
                preactivation += activation[unit] * fixture.residualWeights[row * activation.size() + unit];
            }
            const int64_t residual = std::clamp<int64_t>(preactivation / fixture.residualWeightScale,
                                                        0, fixture.activationScale);
            output += residual * fixture.residualOutput[row];
        }
    }
    const int64_t divisor = static_cast<int64_t>(fixture.activationScale) * fixture.outputScale;
    return static_cast<int>(std::clamp<int64_t>(output / divisor, -32000, 32000));
}

void load(NnueNetwork& network, const FixtureFile& file, const Fixture& fixture, bool legacy = false){
    file.write(encode(fixture, legacy));
    std::string error;
    require(network.load(file.path, &error), "fixture load failed: " + error);
    require(network.formatVersion() == (legacy ? 1U : 2U), "format version getter mismatch");
    require(network.hiddenSize() == fixture.hidden, "hidden size getter mismatch");
    require(network.residualHiddenSize() == (legacy ? 0U : fixture.residualHidden), "residual size getter mismatch");
}

size_t checkOutput(const Fixture& fixture, const FixtureFile& file,
                   NnueNetwork& network, std::mt19937& generator){
    load(network, file, fixture);
    NnueAccumulator accumulator;
    accumulator.valid = true;
    for(auto& values : accumulator.values) values.resize(fixture.hidden);
    Board board;
    board.reset();
    size_t comparisons = 0;
    const std::array<int32_t, 10> extremes{{std::numeric_limits<int32_t>::min(), -32769, -1, 0,
        1, 32767, 32768, fixture.activationScale / 2, fixture.activationScale,
        std::numeric_limits<int32_t>::max()}};
    for(size_t sample = 0; sample < 160; sample++){
        for(size_t perspective = 0; perspective < 2; perspective++){
            for(size_t unit = 0; unit < fixture.hidden; unit++){
                accumulator.values[perspective][unit] = sample < 20 ?
                    extremes[(sample + unit + perspective * 3) % extremes.size()] :
                    static_cast<int32_t>(static_cast<int64_t>(generator()) + std::numeric_limits<int32_t>::min());
            }
        }
        for(size_t stm = 0; stm < 2; stm++){
            board.stm = stm == 0 ? Color::White : Color::Black;
            require(network.evaluate(board, accumulator) == reference(fixture, accumulator, stm),
                    "residual output mismatch, H=" + std::to_string(fixture.hidden) +
                    ", R=" + std::to_string(fixture.residualHidden) + ", sample=" + std::to_string(sample));
            comparisons++;
        }
    }
    return comparisons;
}

void checkRoundingAndSaturation(const FixtureFile& file, NnueNetwork& network){
    Fixture fixture = patterned(1, 1);
    fixture.activationScale = 10;
    fixture.outputScale = 1;
    fixture.outputBias = 0;
    fixture.direct.assign(2, 0);
    fixture.residualWeightScale = 7;
    fixture.residualBias[0] = 13; // 13/7 truncates to 1, not a rounded 2.
    fixture.residualWeights.assign(2, 0);
    fixture.residualOutput[0] = 6;
    NnueAccumulator accumulator;
    accumulator.valid = true;
    accumulator.values[0] = {0};
    accumulator.values[1] = {0};
    Board board;
    board.reset();
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == 0, "residual division must truncate instead of round");
    fixture.outputBias = 6;
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == 1, "branches must combine before final division");
    fixture.residualOutput[0] = 19;
    fixture.outputBias = -28; // Combined numerator -9: C++ truncation gives zero, not -1.
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == 0, "negative final division must truncate toward zero");
    fixture.residualOutput[0] = 15;
    fixture.outputBias = -9;
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == 0, "opposite-sign branches must cancel before division");
    fixture.residualBias[0] = -1;
    fixture.outputBias = 0;
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == 0, "negative residual must clip to zero");
    fixture.residualBias[0] = std::numeric_limits<int32_t>::max();
    fixture.residualOutput[0] = 3;
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == 3, "residual activation must clip to activation scale");
    for(const int32_t bias : {std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min()}){
        fixture.activationScale = 1;
        fixture.residualOutput[0] = 0;
        fixture.outputBias = bias;
        load(network, file, fixture);
        require(network.evaluate(board, accumulator) == (bias > 0 ? 32000 : -32000), "output saturation mismatch");
    }
}

void checkReloadCompatibility(const FixtureFile& file, NnueNetwork& network){
    Fixture fixture = patterned(9, 3);
    NnueAccumulator accumulator;
    accumulator.valid = true;
    accumulator.values[0].assign(9, 12);
    accumulator.values[1].assign(9, 3);
    Board board;
    board.reset();
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == reference(fixture, accumulator, 0), "v2 reload reference mismatch");
    load(network, file, fixture, true);
    require(std::string(network.outputKernelName()).find("residual") == std::string::npos,
            "v2-to-v1 reload retained residual dispatch");
    const int legacyScore = network.evaluate(board, accumulator);
    require(legacyScore == reference(fixture, accumulator, 0, true), "legacy score changed");
    fixture.residualOutput.assign(fixture.residualHidden, 0);
    load(network, file, fixture);
    require(network.evaluate(board, accumulator) == legacyScore, "zero residual v2 must equal legacy output");
    network.clear();
    require(!network.loaded() && network.hiddenSize() == 0 && network.formatVersion() == 0 &&
            network.residualHiddenSize() == 0 && network.source().empty() &&
            std::string(network.outputKernelName()) == "none", "clear retained format/residual state");
    require(network.evaluate(board, accumulator) == 0, "cleared residual network must evaluate to zero");
}

void checkMalformed(const FixtureFile& file, NnueNetwork& network){
    const Fixture fixture = patterned(2, 3);
    const auto valid = encode(fixture);
    const auto rejected = [&](const std::vector<unsigned char>& bytes, const std::string& label){
        load(network, file, fixture); // Ensure failures clear an already loaded v2 network.
        file.write(bytes);
        std::string error;
        require(!network.load(file.path, &error), label + " unexpectedly loaded");
        require(!error.empty(), label + " did not explain failure");
        require(!network.loaded() && network.hiddenSize() == 0 && network.formatVersion() == 0 &&
                network.residualHiddenSize() == 0 && network.source().empty() &&
                std::string(network.outputKernelName()) == "none", label + " retained old network state");
    };
    const std::array<std::pair<size_t, u32>, 15> mutations{{
        {8, 1}, {8, 3}, {12, 40959}, {16, 0}, {16, 1025},
        {20, 0}, {20, static_cast<u32>(-1)}, {24, 0}, {24, static_cast<u32>(-1)},
        {32, 0}, {32, 33}, {32, std::numeric_limits<u32>::max()},
        {36, 0}, {36, static_cast<u32>(-1)}, {12, 0}
    }};
    for(const auto& mutation : mutations){
        auto bytes = valid;
        replace32(bytes, mutation.first, mutation.second);
        rejected(bytes, "invalid header field at " + std::to_string(mutation.first));
    }
    auto bytes = valid;
    bytes[5] = '1';
    rejected(bytes, "v1 magic with v2 version");
    bytes[0] = 'X';
    rejected(bytes, "invalid magic");
    const size_t directStart = 40 + fixture.hidden * sizeof(int32_t) + fixture.input.size() * sizeof(int16_t);
    const size_t residualBiasStart = directStart + fixture.direct.size() * sizeof(int16_t);
    const size_t residualWeightStart = residualBiasStart + fixture.residualBias.size() * sizeof(int32_t);
    const size_t residualOutputStart = residualWeightStart + fixture.residualWeights.size() * sizeof(int16_t);
    for(const size_t size : {size_t{0}, size_t{7}, size_t{8}, size_t{31}, size_t{32}, size_t{35},
                            size_t{39}, size_t{40}, size_t{41}, directStart - 1, directStart + 1,
                            residualBiasStart - 1, residualBiasStart + 1, residualWeightStart + 1,
                            residualOutputStart + 1, valid.size() - 1}){
        rejected(std::vector<unsigned char>(valid.begin(), valid.begin() + static_cast<std::ptrdiff_t>(size)),
                 "truncated payload at " + std::to_string(size));
    }
    bytes = valid;
    bytes.push_back(0);
    rejected(bytes, "trailing payload");

    Fixture boundary = patterned(1, 1);
    boundary.input.assign(NnueNetwork::FeatureCount, 0);
    boundary.input[0] = std::numeric_limits<int16_t>::min();
    boundary.hiddenBias[0] = std::numeric_limits<int32_t>::max() - 64 * 32768;
    load(network, file, boundary); // Equality at INT32_MAX is safe.
    boundary.hiddenBias[0]++;
    rejected(encode(boundary), "unsafe transformer accumulator bound");
    boundary.input[0] = 0;
    boundary.hiddenBias[0] = std::numeric_limits<int32_t>::min();
    rejected(encode(boundary), "INT32_MIN transformer absolute bound");
    load(network, file, boundary, true); // Preserve legacy v1 acceptance semantics.
}

Move findMove(Board& board, const std::string& uci){
    MoveList legal;
    board.genLegalMoves(legal);
    for(const Move& move : legal) if(moveToUCI(move) == uci) return move;
    throw std::runtime_error("fixture legal move missing: " + uci);
}

void checkIncremental(const FixtureFile& file, NnueNetwork& network, u32 hidden){
    const Fixture fixture = patterned(hidden, 5);
    load(network, file, fixture);
    Zobrist zobrist;
    const auto compare = [&](const Board& board, const NnueAccumulator& incremental){
        NnueAccumulator rebuilt;
        network.refresh(board, rebuilt);
        require(incremental.valid && rebuilt.valid && incremental.values == rebuilt.values,
                "v2 incremental accumulator mismatch");
        const size_t stm = board.stm == Color::White ? 0 : 1;
        require(network.evaluate(board, incremental) == reference(fixture, rebuilt, stm),
                "v2 incremental output mismatch");
        require(network.evaluate(board) == network.evaluate(board, incremental), "v2 rebuild output mismatch");
    };
    const auto exercise = [&](const std::string& fen, const std::string& uci){
        Board board;
        board.setZobrist(&zobrist);
        require(board.loadFEN(fen), "incremental fixture FEN rejected");
        NnueAccumulator parent;
        network.refresh(board, parent);
        const Move move = findMove(board, uci);
        Undo undo{};
        require(board.makeMove(move, undo), "incremental fixture move rejected");
        NnueAccumulator child;
        network.updateAfterMove(board, undo, parent, child);
        compare(board, child);
        NnueAccumulator inPlace = parent;
        network.updateAfterMove(board, undo, inPlace, inPlace);
        compare(board, inPlace);
        NnueAccumulator invalid;
        network.updateAfterMove(board, undo, invalid, child);
        compare(board, child);
        invalid.valid = true;
        invalid.values[0].resize(hidden + 1);
        invalid.values[1].resize(hidden);
        network.updateAfterMove(board, undo, invalid, child);
        compare(board, child);
        invalid.values[0].resize(hidden);
        invalid.values[1].clear();
        network.updateAfterMove(board, undo, invalid, child);
        compare(board, child);
        board.undoMove(undo);
        compare(board, parent);
    };
    exercise("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e4");
    exercise("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1", "e7e5");
    exercise("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5");
    exercise("4k3/8/8/3p4/4P3/8/8/4K3 b - - 0 1", "d5e4");
    exercise("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6");
    exercise("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1", "d4e3");
    for(const std::string uci : {"e1g1", "e1c1", "e8g8", "e8c8"}){
        exercise("r3k2r/8/8/8/8/8/8/R3K2R " + std::string(uci[1] == '1' ? "w" : "b") +
                 " KQkq - 0 1", uci);
    }
    for(const char promotion : {'q', 'r', 'b', 'n'}){
        exercise("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", std::string("a7a8") + promotion);
        exercise("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", std::string("a7b8") + promotion);
        exercise("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", std::string("a2a1") + promotion);
        exercise("4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1", std::string("a2b1") + promotion);
    }
    exercise("4k3/8/8/8/8/8/8/4K3 w - - 0 1", "e1d1");
    exercise("4k3/8/8/8/8/8/8/4K3 b - - 0 1", "e8d8");
    exercise("4k3/8/8/8/8/3p4/4K3/8 w - - 0 1", "e2d3");
    exercise("8/4k3/3P4/8/8/8/8/4K3 b - - 0 1", "e7d6");

    Board board;
    board.setZobrist(&zobrist);
    board.reset();
    NnueAccumulator root;
    network.refresh(board, root);
    std::vector<NnueAccumulator> stack{root};
    std::vector<Undo> undos;
    std::mt19937 random(0x52455349U + hidden);
    for(size_t ply = 0; ply < 96; ply++){
        MoveList legal;
        board.genLegalMoves(legal);
        if(legal.empty()) break;
        Undo undo{};
        require(board.makeMove(legal[static_cast<size_t>(random() % legal.size())], undo), "random move rejected");
        NnueAccumulator child;
        network.updateAfterMove(board, undo, stack.back(), child);
        compare(board, child);
        undos.push_back(undo);
        stack.push_back(std::move(child));
    }
    while(!undos.empty()){
        board.undoMove(undos.back());
        undos.pop_back();
        stack.pop_back();
        compare(board, stack.back());
    }
    NullUndo undo{};
    board.makeNullMove(undo);
    NnueAccumulator copied;
    network.copyAccumulator(stack.back(), copied);
    compare(board, copied);
    board.undoNullMove(undo);
}
}

int main(){
    try {
        FixtureFile file;
        NnueNetwork network;
        std::mt19937 generator(0x56325245U);
        size_t fixtures = 0;
        size_t comparisons = 0;
        const auto check = [&](const Fixture& fixture){
            comparisons += checkOutput(fixture, file, network, generator);
            fixtures++;
        };
        for(const u32 hidden : {1U, 4U, 7U, 8U, 9U, 17U}){
            for(const u32 residual : {1U, 3U, 8U, 32U}) check(patterned(hidden, residual));
        }
        for(const int32_t scale : {1, 32767, 32768, std::numeric_limits<int32_t>::max()}){
            Fixture fixture = patterned(9, 3);
            fixture.activationScale = scale;
            fixture.outputScale = std::numeric_limits<int32_t>::max();
            fixture.outputBias = std::numeric_limits<int32_t>::min();
            fixture.residualWeightScale = std::numeric_limits<int32_t>::max();
            fixture.residualBias.assign(3, std::numeric_limits<int32_t>::min());
            fixture.residualWeights.assign(fixture.residualWeights.size(), std::numeric_limits<int16_t>::min());
            fixture.direct.assign(fixture.direct.size(), std::numeric_limits<int16_t>::max());
            fixture.residualOutput.assign(3, std::numeric_limits<int16_t>::min());
            check(fixture);
            fixture.residualBias.assign(3, std::numeric_limits<int32_t>::max());
            fixture.residualWeights.assign(fixture.residualWeights.size(), std::numeric_limits<int16_t>::max());
            fixture.direct.assign(fixture.direct.size(), std::numeric_limits<int16_t>::min());
            check(fixture);
        }
        // Do not let a large final divisor hide overflow in the residual dot.
        // Its exact wide numerator clips to A, giving a small, unsaturated 2cp.
        Fixture wide = patterned(9, 1);
        wide.activationScale = std::numeric_limits<int32_t>::max();
        wide.outputScale = 1;
        wide.outputBias = 0;
        wide.direct.assign(wide.direct.size(), 0);
        wide.residualWeightScale = 1;
        wide.residualBias[0] = 0;
        wide.residualWeights.assign(wide.residualWeights.size(), std::numeric_limits<int16_t>::max());
        wide.residualOutput[0] = 2;
        check(wide);
        checkRoundingAndSaturation(file, network);
        checkReloadCompatibility(file, network);
        checkMalformed(file, network);
        for(const u32 hidden : {1U, 8U, 17U}) checkIncremental(file, network, hidden);
        std::cout << "NNUE residual tests passed: " << fixtures << " numeric fixtures, " << comparisons
                  << " independent-reference comparisons, format/reload/incremental checks\n";
        return 0;
    } catch(const std::exception& exception){
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
