#include "nnue.hpp"

#include <limits>
#include <stdexcept>

namespace {
struct Fixture {
    u32 hidden;
    int32_t scale;
    int32_t outputScale;
    int32_t bias;
    std::vector<int16_t> weights;
};

void require(bool condition, const std::string& message){
    if(!condition) throw std::runtime_error(message);
}

// Owned temporary artifact: never overwrite an existing user file.
struct FixtureFile {
    std::filesystem::path path;
    FixtureFile(){
        const auto stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() /
            ("forklift-nnue-output-test-" + std::to_string(stamp) + "-" +
             std::to_string(std::random_device{}()) + ".nnue");
        require(!std::filesystem::exists(path), "temporary filename collision");
    }
    ~FixtureFile(){
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    void write(const Fixture& fixture) const {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        const std::array<char, 8> magic{{'T','N','N','U','E','1','\0','\0'}};
        const u32 version = NnueNetwork::FormatVersion;
        const u32 features = NnueNetwork::FeatureCount;
        output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        output.write(reinterpret_cast<const char*>(&version), sizeof(version));
        output.write(reinterpret_cast<const char*>(&features), sizeof(features));
        output.write(reinterpret_cast<const char*>(&fixture.hidden), sizeof(fixture.hidden));
        output.write(reinterpret_cast<const char*>(&fixture.scale), sizeof(fixture.scale));
        output.write(reinterpret_cast<const char*>(&fixture.outputScale), sizeof(fixture.outputScale));
        output.write(reinterpret_cast<const char*>(&fixture.bias), sizeof(fixture.bias));
        std::vector<int32_t> bias(fixture.hidden);
        for(size_t unit = 0; unit < bias.size(); unit++) bias[unit] = static_cast<int32_t>(unit % 11) - 5;
        output.write(reinterpret_cast<const char*>(bias.data()),
                     static_cast<std::streamsize>(bias.size() * sizeof(int32_t)));
        const std::array<char, 65536> zero{};
        size_t remaining = static_cast<size_t>(features) * fixture.hidden * sizeof(int16_t);
        while(remaining != 0){
            const size_t bytes = std::min(remaining, zero.size());
            output.write(zero.data(), static_cast<std::streamsize>(bytes));
            remaining -= bytes;
        }
        output.write(reinterpret_cast<const char*>(fixture.weights.data()),
                     static_cast<std::streamsize>(fixture.weights.size() * sizeof(int16_t)));
        require(static_cast<bool>(output), "could not write test fixture");
    }
};

int64_t referenceOutput(const Fixture& fixture, const NnueAccumulator& accumulator,
                        size_t perspective){
    int64_t output = fixture.bias;
    for(size_t unit = 0; unit < fixture.hidden; unit++){
        const int32_t first = std::clamp(accumulator.values[perspective][unit], 0, fixture.scale);
        const int32_t second = std::clamp(accumulator.values[1 - perspective][unit], 0, fixture.scale);
        output += static_cast<int64_t>(first) * fixture.weights[unit];
        output += static_cast<int64_t>(second) * fixture.weights[fixture.hidden + unit];
    }
    return output;
}

int referenceScore(const Fixture& fixture, const NnueAccumulator& accumulator, size_t perspective){
    const int64_t divisor = static_cast<int64_t>(fixture.scale) * fixture.outputScale;
    return static_cast<int>(std::clamp<int64_t>(referenceOutput(fixture, accumulator, perspective) / divisor,
                                               -32000, 32000));
}

const char* expectedKernel(const Fixture& fixture){
    int64_t bound = std::abs(static_cast<int64_t>(fixture.bias));
    for(const int16_t weight : fixture.weights){
        bound += static_cast<int64_t>(fixture.scale) * std::abs(static_cast<int32_t>(weight));
    }
    if(bound > std::numeric_limits<int32_t>::max()) return "scalar64";
#if !defined(FORKLIFT_NNUE_FORCE_SCALAR) && !defined(_M_ARM64EC) && !defined(__arm64ec__) && \
    (defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
    if(fixture.scale <= std::numeric_limits<int16_t>::max() && fixture.hidden >= 8) return "sse2";
#endif
    return "scalar32";
}

size_t checkFixture(const Fixture& fixture, const FixtureFile& file,
                    NnueNetwork& network, std::mt19937& generator){
    file.write(fixture);
    std::string error;
    require(network.load(file.path, &error), "network load: " + error);
    require(std::string(network.outputKernelName()) == expectedKernel(fixture), "incorrect output kernel dispatch");
    Board board;
    board.reset();
    NnueAccumulator accumulator;
    accumulator.valid = true;
    for(auto& values : accumulator.values) values.resize(fixture.hidden);
    size_t comparisons = 0;
    const std::array<int32_t, 11> extremes{{std::numeric_limits<int32_t>::min(), -32769, -32768,
        -1, 0, 1, 32766, 32767, 32768, fixture.scale, std::numeric_limits<int32_t>::max()}};
    for(size_t sample = 0; sample < 1024; sample++){
        for(size_t perspective = 0; perspective < 2; perspective++){
            for(size_t unit = 0; unit < fixture.hidden; unit++){
                if(sample == 0) accumulator.values[perspective][unit] = std::numeric_limits<int32_t>::max();
                else if(sample == 1) accumulator.values[perspective][unit] = std::numeric_limits<int32_t>::min();
                else if(sample < 16) accumulator.values[perspective][unit] = extremes[(sample + unit + perspective) % extremes.size()];
                else {
                    // Conversion from a representable int64 value, avoiding
                    // implementation-defined uint32-to-int32 narrowing.
                    accumulator.values[perspective][unit] = static_cast<int32_t>(
                        static_cast<int64_t>(generator()) + std::numeric_limits<int32_t>::min());
                }
            }
        }
        for(size_t perspective = 0; perspective < 2; perspective++){
            board.stm = perspective == 0 ? Color::White : Color::Black;
            require(network.evaluate(board, accumulator) == referenceScore(fixture, accumulator, perspective),
                    "output mismatch, width=" + std::to_string(fixture.hidden) + ", scale=" +
                    std::to_string(fixture.scale) + ", sample=" + std::to_string(sample));
            comparisons++;
        }
    }

    // An invalid accumulator or invalid parent must rebuild rather than enter
    // an output kernel with missing lanes. All fixture input weights are zero.
    NnueAccumulator refreshed;
    network.refresh(board, refreshed);
    require(refreshed.valid, "refresh should produce valid accumulator");
    const size_t perspective = board.stm == Color::White ? 0 : 1;
    const int expected = referenceScore(fixture, refreshed, perspective);
    NnueAccumulator invalid;
    require(network.evaluate(board, invalid) == expected, "invalid accumulator fallback mismatch");
    invalid.valid = true;
    require(network.evaluate(board, invalid) == expected, "wrong-sized accumulator fallback mismatch");
    Move move{6, 21}; // g1f3, same state for each fixture.
    board.stm = Color::White;
    Undo undo;
    require(board.makeMove(move, undo), "fixture move failed");
    NnueAccumulator rebuilt;
    network.refresh(board, rebuilt);
    NnueAccumulator child;
    network.updateAfterMove(board, undo, invalid, child);
    require(child.valid && child.values == rebuilt.values, "invalid parent rebuild mismatch");
    require(network.evaluate(board, child) == referenceScore(fixture, rebuilt, 1), "rebuilt child output mismatch");
    invalid = NnueAccumulator{};
    network.updateAfterMove(board, undo, invalid, child);
    require(child.valid && child.values == rebuilt.values, "invalid parent flag rebuild mismatch");

    // Stale certificates must not survive clear, a failed load, or loading a
    // later fixture whose scale/bias/weights choose a different kernel.
    network.clear();
    require(!network.loaded() && std::string(network.outputKernelName()) == "none", "clear retained output dispatch");
    require(network.evaluate(board, child) == 0, "cleared network should return zero");
    require(network.load(file.path, &error), "reload failed");
    require(std::string(network.outputKernelName()) == expectedKernel(fixture), "reload dispatch mismatch");
    require(network.evaluate(board, child) == referenceScore(fixture, rebuilt, 1), "reload output mismatch");
    require(!network.load(file.path.string() + ".missing", &error), "nonexistent file unexpectedly loaded");
    require(!network.loaded() && std::string(network.outputKernelName()) == "none", "failed load retained dispatch");
    return comparisons;
}

Fixture patterned(u32 hidden, int32_t scale){
    Fixture fixture{hidden, scale, 3, 11, std::vector<int16_t>(static_cast<size_t>(hidden) * 2)};
    for(size_t unit = 0; unit < fixture.weights.size(); unit++){
        fixture.weights[unit] = static_cast<int16_t>(static_cast<int>(unit * 7 % 31) - 15);
    }
    return fixture;
}
}

int main(){
    try {
        FixtureFile file;
        NnueNetwork network;
        std::mt19937 generator(0x53494D44U);
        size_t fixtures = 0;
        size_t comparisons = 0;
        const auto check = [&](const Fixture& fixture){
            comparisons += checkFixture(fixture, file, network, generator);
            fixtures++;
        };
        for(const u32 width : {1U, 4U, 7U, 8U, 9U, 17U, 255U, 256U, 257U, 1024U}){
            check(patterned(width, 1024));
        }
        for(const int32_t scale : {1, 32767, 32768, std::numeric_limits<int32_t>::max()}){
            Fixture fixture = patterned(17, scale);
            check(fixture);
            fixture.weights.assign(34, 0);
            fixture.bias = 0;
            fixture.weights[0] = std::numeric_limits<int16_t>::min();
            check(fixture);
            fixture.weights[0] = 1;
            check(fixture);
        }
        for(const int32_t bias : {std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()}){
            Fixture fixture{17, 1024, std::numeric_limits<int32_t>::max(), bias, std::vector<int16_t>(34, 0)};
            check(fixture);
            fixture.weights[0] = 1;
            check(fixture);
        }
        // 2048 * 32767 * 32 + 65535 == INT32_MAX exactly. Moving the
        // bias one unit past it must dispatch to the wide reference kernel.
        Fixture boundary{1024, 32, 32768, 65535, std::vector<int16_t>(2048, 32767)};
        check(boundary);
        boundary.bias++;
        check(boundary);
        boundary.weights.assign(2048, -32768);
        boundary.bias = 0; // abs-bound == INT32_MAX + 1.
        check(boundary);
        boundary.weights[0] = -32767; // abs-bound == INT32_MAX - 31.
        check(boundary);
        // A tail-only nonzero coefficient catches dropped/misordered lanes
        // and tests distinct STM output-weight halves for odd hidden widths.
        Fixture tail{17, 1, 1, 0, std::vector<int16_t>(34, 0)};
        tail.weights[16] = -32768;
        tail.weights[33] = 32767;
        check(tail);
        std::cout << "NNUE output exactness passed: " << fixtures << " fixtures, " << comparisons
                  << " scalar-reference comparisons, dispatch/fallback/reload checks\n";
        return 0;
    } catch(const std::exception& exception){
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
