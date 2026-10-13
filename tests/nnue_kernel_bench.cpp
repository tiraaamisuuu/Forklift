#include "nnue.hpp"

#include <limits>
#include <stdexcept>

#if !defined(_M_ARM64EC) && !defined(__arm64ec__) && \
    (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#include <emmintrin.h>
#define NNUE_BENCH_SSE2 1
#endif

// Isolated inference microbenchmark, not a playing-strength measurement. Build
// with the same optimization flags as the engine and pass a real .nnue file.
namespace {
struct OutputFixture {
    u32 hidden = 0;
    int32_t scale = 0;
    int32_t outputScale = 0;
    int32_t bias = 0;
    std::vector<int16_t> weights;
};

OutputFixture readOutput(const std::filesystem::path& path){
    std::ifstream input(path, std::ios::binary);
    OutputFixture fixture;
    u32 version = 0;
    u32 features = 0;
    input.seekg(8);
    input.read(reinterpret_cast<char*>(&version), sizeof(version));
    input.read(reinterpret_cast<char*>(&features), sizeof(features));
    input.read(reinterpret_cast<char*>(&fixture.hidden), sizeof(fixture.hidden));
    input.read(reinterpret_cast<char*>(&fixture.scale), sizeof(fixture.scale));
    input.read(reinterpret_cast<char*>(&fixture.outputScale), sizeof(fixture.outputScale));
    input.read(reinterpret_cast<char*>(&fixture.bias), sizeof(fixture.bias));
    if(!input || version != NnueNetwork::FormatVersion || features != NnueNetwork::FeatureCount ||
       fixture.hidden == 0 || fixture.hidden > 1024 || fixture.scale <= 0 || fixture.outputScale <= 0){
        throw std::runtime_error("invalid NNUE header");
    }
    const auto offset = static_cast<std::streamoff>(32 + fixture.hidden * sizeof(int32_t) +
        static_cast<size_t>(features) * fixture.hidden * sizeof(int16_t));
    input.seekg(offset);
    fixture.weights.resize(static_cast<size_t>(fixture.hidden) * 2);
    input.read(reinterpret_cast<char*>(fixture.weights.data()),
               static_cast<std::streamsize>(fixture.weights.size() * sizeof(int16_t)));
    if(!input) throw std::runtime_error("truncated output weights");
    return fixture;
}

#if defined(_MSC_VER)
#define NNUE_BENCH_NOINLINE __declspec(noinline)
#else
#define NNUE_BENCH_NOINLINE __attribute__((noinline))
#endif

NNUE_BENCH_NOINLINE int64_t reference(const OutputFixture& fixture,
                                     const NnueAccumulator& accumulator, size_t perspective){
    int64_t sum = fixture.bias;
    for(size_t unit = 0; unit < fixture.hidden; unit++){
        const int32_t first = std::clamp(accumulator.values[perspective][unit], 0, fixture.scale);
        const int32_t second = std::clamp(accumulator.values[1 - perspective][unit], 0, fixture.scale);
        sum += static_cast<int64_t>(first) * fixture.weights[unit];
        sum += static_cast<int64_t>(second) * fixture.weights[fixture.hidden + unit];
    }
    return sum;
}

NNUE_BENCH_NOINLINE int64_t split64(const OutputFixture& fixture,
                                   const NnueAccumulator& accumulator, size_t perspective){
    int64_t firstSum = fixture.bias;
    int64_t secondSum = 0;
    for(size_t unit = 0; unit < fixture.hidden; unit++){
        const int32_t first = std::clamp(accumulator.values[perspective][unit], 0, fixture.scale);
        const int32_t second = std::clamp(accumulator.values[1 - perspective][unit], 0, fixture.scale);
        firstSum += static_cast<int64_t>(first) * fixture.weights[unit];
        secondSum += static_cast<int64_t>(second) * fixture.weights[fixture.hidden + unit];
    }
    return firstSum + secondSum;
}

NNUE_BENCH_NOINLINE int64_t bounded32(const OutputFixture& fixture,
                                     const NnueAccumulator& accumulator, size_t perspective){
    int32_t sum = fixture.bias;
    for(size_t unit = 0; unit < fixture.hidden; unit++){
        const int32_t first = std::clamp(accumulator.values[perspective][unit], 0, fixture.scale);
        const int32_t second = std::clamp(accumulator.values[1 - perspective][unit], 0, fixture.scale);
        sum += first * fixture.weights[unit];
        sum += second * fixture.weights[fixture.hidden + unit];
    }
    return sum;
}

NNUE_BENCH_NOINLINE int64_t split32(const OutputFixture& fixture,
                                   const NnueAccumulator& accumulator, size_t perspective){
    int32_t firstSum = fixture.bias;
    int32_t secondSum = 0;
    for(size_t unit = 0; unit < fixture.hidden; unit++){
        const int32_t first = std::clamp(accumulator.values[perspective][unit], 0, fixture.scale);
        const int32_t second = std::clamp(accumulator.values[1 - perspective][unit], 0, fixture.scale);
        firstSum += first * fixture.weights[unit];
        secondSum += second * fixture.weights[fixture.hidden + unit];
    }
    return static_cast<int64_t>(firstSum) + secondSum;
}

#if defined(NNUE_BENCH_SSE2)
NNUE_BENCH_NOINLINE int64_t packedSse2(const OutputFixture& fixture,
                                     const NnueAccumulator& accumulator, size_t perspective){
    const __m128i zero = _mm_setzero_si128();
    const __m128i ceiling = _mm_set1_epi16(static_cast<int16_t>(fixture.scale));
    __m128i sum = zero;
    size_t unit = 0;
    for(; unit + 8 <= fixture.hidden; unit += 8){
        const auto clipped = [&](const int32_t* values){
            const __m128i low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
            const __m128i high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + 4));
            return _mm_min_epi16(_mm_max_epi16(_mm_packs_epi32(low, high), zero), ceiling);
        };
        const __m128i first = clipped(accumulator.values[perspective].data() + unit);
        const __m128i second = clipped(accumulator.values[1 - perspective].data() + unit);
        const __m128i firstWeights = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(fixture.weights.data() + unit));
        const __m128i secondWeights = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(fixture.weights.data() + fixture.hidden + unit));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(first, firstWeights));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(second, secondWeights));
    }
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
    sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
    int64_t output = static_cast<int64_t>(_mm_cvtsi128_si32(sum)) + fixture.bias;
    for(; unit < fixture.hidden; unit++){
        const int32_t first = std::clamp(accumulator.values[perspective][unit], 0, fixture.scale);
        const int32_t second = std::clamp(accumulator.values[1 - perspective][unit], 0, fixture.scale);
        output += first * fixture.weights[unit];
        output += second * fixture.weights[fixture.hidden + unit];
    }
    return output;
}
#endif

template<class Function>
void measure(const char* name, Function&& function, const std::vector<NnueAccumulator>& pool,
             size_t iterations){
    uint64_t checksum = 0;
    const auto start = std::chrono::steady_clock::now();
    for(size_t sample = 0; sample < iterations; sample++){
        checksum += static_cast<uint64_t>(function(pool[sample % pool.size()], sample % 2));
    }
    const double elapsed = std::chrono::duration<double, std::nano>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << name << "," << elapsed / static_cast<double>(iterations) << "," << checksum << '\n';
}
}

int main(int argc, char** argv){
    try {
        if(argc < 2 || argc > 3){
            std::cerr << "usage: nnue-kernel-bench NETWORK [ITERATIONS]\n";
            return 1;
        }
        const size_t iterations = argc == 3 ? std::stoull(argv[2]) : 1000000;
        if(iterations == 0 || iterations > 100000000) throw std::runtime_error("invalid iterations");
        const OutputFixture fixture = readOutput(argv[1]);
        NnueNetwork network;
        std::string error;
        if(!network.load(argv[1], &error)) throw std::runtime_error(error);
        int64_t bound = std::abs(static_cast<int64_t>(fixture.bias));
        for(const int16_t weight : fixture.weights){
            bound += static_cast<int64_t>(fixture.scale) * std::abs(static_cast<int32_t>(weight));
        }
        const bool safe32 = bound <= std::numeric_limits<int32_t>::max();
        const bool safePacked = safe32 && fixture.scale <= std::numeric_limits<int16_t>::max();
        std::cout << "hidden=" << fixture.hidden << ",scale=" << fixture.scale
                  << ",absolute_bound=" << bound << ",safe32=" << safe32
                  << ",network_kernel=" << network.outputKernelName() << '\n';
        std::mt19937 generator(0x4E4E5545U);
        std::vector<NnueAccumulator> pool(128);
        for(auto& accumulator : pool){
            accumulator.valid = true;
            for(auto& values : accumulator.values){
                values.resize(fixture.hidden);
                for(int32_t& value : values){
                    const int64_t range = static_cast<int64_t>(fixture.scale) * 3 + 1;
                    value = static_cast<int32_t>(std::clamp<int64_t>(
                        static_cast<int64_t>(generator()) % range - fixture.scale,
                        std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max()));
                }
            }
            for(size_t perspective = 0; perspective < 2; perspective++){
                const int64_t expected = reference(fixture, accumulator, perspective);
                if(expected != split64(fixture, accumulator, perspective) ||
                   (safe32 && (expected != bounded32(fixture, accumulator, perspective) ||
                               expected != split32(fixture, accumulator, perspective)))){
                    throw std::runtime_error("kernel exactness failure");
                }
#if defined(NNUE_BENCH_SSE2)
                if(safePacked && expected != packedSse2(fixture, accumulator, perspective)){
                    throw std::runtime_error("SSE2 kernel exactness failure");
                }
#endif
                Board board;
                board.stm = perspective == 0 ? Color::White : Color::Black;
                const int64_t divisor = static_cast<int64_t>(fixture.scale) * fixture.outputScale;
                const int expectedScore = static_cast<int>(std::clamp<int64_t>(expected / divisor, -32000, 32000));
                if(network.evaluate(board, accumulator) != expectedScore){
                    throw std::runtime_error("public inference exactness failure");
                }
            }
        }
        std::cout << "kernel,ns_per_call,checksum\n";
        for(int round = 0; round < 5; round++){
            measure("reference64", [&](const auto& a, size_t p){ return reference(fixture, a, p); }, pool, iterations);
            measure("split64", [&](const auto& a, size_t p){ return split64(fixture, a, p); }, pool, iterations);
            if(safe32){
                measure("bounded32", [&](const auto& a, size_t p){ return bounded32(fixture, a, p); }, pool, iterations);
                measure("split32", [&](const auto& a, size_t p){ return split32(fixture, a, p); }, pool, iterations);
            }
#if defined(NNUE_BENCH_SSE2)
            if(safePacked){
                measure("packedSse2", [&](const auto& a, size_t p){ return packedSse2(fixture, a, p); }, pool, iterations);
            }
#else
            (void)safePacked;
#endif
            measure("network", [&](const auto& a, size_t p){
                Board board;
                board.stm = p == 0 ? Color::White : Color::Black;
                return network.evaluate(board, a);
            }, pool, iterations);
        }
        return 0;
    } catch(const std::exception& exception){
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
