#pragma once

#include "board.hpp"

#include <limits>

// SSE2 is part of the x64 baseline, not an AVX/native-architecture requirement.
// Other targets retain the portable kernels. The override allows the scalar
// implementation to be checked by the same tests in a separate build.
#if !defined(FORKLIFT_NNUE_FORCE_SCALAR) && !defined(_M_ARM64EC) && !defined(__arm64ec__) && \
    (defined(__SSE2__) || defined(_M_X64) || \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
#include <emmintrin.h>
#define FORKLIFT_NNUE_HAS_SSE2 1
#endif

struct NnueAccumulator {
    std::array<std::vector<int32_t>, 2> values;
    bool valid = false;
};

class NnueNetwork {
public:
    static constexpr u32 FormatVersion = 1;
    static constexpr u32 FeatureCount = 64U * 10U * 64U; // HalfKP: king, relative piece, square.

    bool load(const std::filesystem::path& path, std::string* error = nullptr){
        auto fail = [&](const std::string& message){
            clear();
            if(error) *error = message;
            return false;
        };

        std::ifstream input(path, std::ios::binary);
        if(!input) return fail("could not open network file");

        std::array<char, 8> magic{};
        input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
        const std::array<char, 8> expected{{'T','N','N','U','E','1','\0','\0'}};
        if(magic != expected) return fail("invalid NNUE magic");

        u32 version = 0;
        u32 features = 0;
        u32 hidden = 0;
        int32_t hiddenScale = 0;
        int32_t outputScale = 0;
        int32_t outputBias = 0;
        if(!readValue(input, version) || !readValue(input, features) || !readValue(input, hidden) ||
           !readValue(input, hiddenScale) || !readValue(input, outputScale) || !readValue(input, outputBias)){
            return fail("truncated NNUE header");
        }
        if(version != FormatVersion) return fail("unsupported NNUE version");
        if(features != FeatureCount) return fail("network feature layout does not match HalfKP v1");
        if(hidden == 0 || hidden > 1024) return fail("invalid NNUE hidden size");
        if(hiddenScale <= 0 || outputScale <= 0) return fail("invalid NNUE quantization scale");

        const size_t inputWeightCount = static_cast<size_t>(features) * hidden;
        const size_t outputWeightCount = static_cast<size_t>(hidden) * 2;
        std::vector<int32_t> newHiddenBias(hidden);
        std::vector<int16_t> newInputWeights(inputWeightCount);
        std::vector<int16_t> newOutputWeights(outputWeightCount);
        if(!readArray(input, newHiddenBias) || !readArray(input, newInputWeights) ||
           !readArray(input, newOutputWeights)){
            return fail("truncated NNUE weights");
        }

        hiddenSize_ = hidden;
        hiddenScale_ = hiddenScale;
        outputScale_ = outputScale;
        outputBias_ = outputBias;
        hiddenBias_ = std::move(newHiddenBias);
        inputWeights_ = std::move(newInputWeights);
        outputWeights_ = std::move(newOutputWeights);
        // Every clipped activation lies in [0, hiddenScale]. Certify a bound
        // for every product and every partial reduction, not just the final
        // score. The certificate is independent of the board/accumulator.
        int64_t absoluteBound = std::abs(static_cast<int64_t>(outputBias_));
        for(const int16_t weight : outputWeights_){
            absoluteBound += static_cast<int64_t>(hiddenScale_) *
                std::abs(static_cast<int32_t>(weight));
        }
        boundedOutput32_ = absoluteBound <= std::numeric_limits<int32_t>::max();
        source_ = path.string();
        return true;
    }

    void clear(){
        hiddenSize_ = 0;
        hiddenScale_ = 0;
        outputScale_ = 0;
        outputBias_ = 0;
        boundedOutput32_ = false;
        hiddenBias_.clear();
        inputWeights_.clear();
        outputWeights_.clear();
        source_.clear();
    }

    bool loaded() const { return hiddenSize_ != 0; }
    u32 hiddenSize() const { return hiddenSize_; }
    const std::string& source() const { return source_; }
    const char* outputKernelName() const {
        if(!loaded()) return "none";
#if defined(FORKLIFT_NNUE_FORCE_REFERENCE)
        return "scalar64-reference";
#else
#if defined(FORKLIFT_NNUE_HAS_SSE2)
        if(boundedOutput32_ && hiddenScale_ <= std::numeric_limits<int16_t>::max() && hiddenSize_ >= 8){
            return "sse2";
        }
#endif
        return boundedOutput32_ ? "scalar32" : "scalar64";
#endif
    }

    int evaluate(const Board& board) const {
        if(!loaded()) return 0;
        thread_local NnueAccumulator accumulator;
        refresh(board, accumulator);
        if(!accumulator.valid) return 0;
        return evaluate(board, accumulator);
    }

    int evaluate(const Board& board, const NnueAccumulator& accumulator) const {
        if(!loaded() || !accumulator.valid) return evaluate(board);
        const auto& firstAccumulator = accumulator.values[colorIndex(board.stm)];
        const auto& secondAccumulator = accumulator.values[colorIndex(other(board.stm))];
        if(firstAccumulator.size() != hiddenSize_ || secondAccumulator.size() != hiddenSize_){
            return evaluate(board);
        }

        int64_t output;
#if defined(FORKLIFT_NNUE_FORCE_REFERENCE)
        output = outputScalar<int64_t>(firstAccumulator.data(), secondAccumulator.data());
#else
        if(boundedOutput32_){
#if defined(FORKLIFT_NNUE_HAS_SSE2)
            if(hiddenScale_ <= std::numeric_limits<int16_t>::max() && hiddenSize_ >= 8){
                output = outputSse2(firstAccumulator.data(), secondAccumulator.data());
            } else
#endif
            {
                output = outputScalar<int32_t>(firstAccumulator.data(), secondAccumulator.data());
            }
        } else {
            output = outputScalar<int64_t>(firstAccumulator.data(), secondAccumulator.data());
        }
#endif
        const int64_t divisor = static_cast<int64_t>(hiddenScale_) * outputScale_;
        return static_cast<int>(std::clamp<int64_t>(output / divisor, -32000, 32000));
    }

    void refresh(const Board& board, NnueAccumulator& accumulator) const {
        if(!loaded() || board.findKing(Color::White) < 0 || board.findKing(Color::Black) < 0){
            accumulator.valid = false;
            return;
        }
        refreshPerspective(board, Color::White, accumulator.values[0]);
        refreshPerspective(board, Color::Black, accumulator.values[1]);
        accumulator.valid = true;
    }

    void updateAfterMove(const Board& board, const Undo& undo,
                         const NnueAccumulator& parent, NnueAccumulator& child) const {
        if(!loaded() || !parent.valid || parent.values[0].size() != hiddenSize_ ||
           parent.values[1].size() != hiddenSize_){
            refresh(board, child);
            return;
        }
        child.valid = true;

        const Move& move = undo.m;
        const Piece resultingPiece = board.b[move.to];
        Piece movingPiece = resultingPiece;
        if(move.promo != PieceType::None) movingPiece.t = PieceType::Pawn;
        const bool kingMoved = movingPiece.t == PieceType::King;
        const int captureSquare = move.isEnPassant
            ? int(move.to) + (movingPiece.c == Color::White ? -8 : 8)
            : int(move.to);

        for(const Color perspective : {Color::White, Color::Black}){
            const size_t index = colorIndex(perspective);
            auto& values = child.values[index];
            if(kingMoved && movingPiece.c == perspective){
                refreshPerspective(board, perspective, values);
                continue;
            }

            if(!kingMoved){
                // Write each lane once: copy the parent and apply all feature deltas
                // together. Promotions and en passant use the same kernels, with
                // the pre-move pawn and the actual captured square respectively.
                const auto& source = parent.values[index];
                const int16_t* removed = featureWeights(board, perspective, move.from, movingPiece);
                const int16_t* added = featureWeights(board, perspective, move.to, resultingPiece);
                values.resize(hiddenSize_);
                if(!isNone(undo.captured) && undo.captured.t != PieceType::King){
                    const int16_t* captured = featureWeights(board, perspective, captureSquare, undo.captured);
                    for(size_t hidden = 0; hidden < hiddenSize_; hidden++){
                        values[hidden] = source[hidden] - removed[hidden] - captured[hidden] + added[hidden];
                    }
                } else {
                    for(size_t hidden = 0; hidden < hiddenSize_; hidden++){
                        values[hidden] = source[hidden] - removed[hidden] + added[hidden];
                    }
                }
                continue;
            }

            // A king move changes its own HalfKP bucket (refreshed above). For
            // the opposite perspective, retain the capture/castling delta path.
            values = parent.values[index];
            if(!isNone(undo.captured) && undo.captured.t != PieceType::King){
                addFeature(board, perspective, captureSquare, undo.captured, -1, values);
            }
            if(move.isCastle){
                const int rookFrom = move.to > move.from
                    ? (movingPiece.c == Color::White ? 7 : 63)
                    : (movingPiece.c == Color::White ? 0 : 56);
                const int rookTo = move.to > move.from ? int(move.to) - 1 : int(move.to) + 1;
                const Piece rook{PieceType::Rook, movingPiece.c};
                addFeature(board, perspective, rookFrom, rook, -1, values);
                addFeature(board, perspective, rookTo, rook, 1, values);
            }
        }
    }

    void copyAccumulator(const NnueAccumulator& source, NnueAccumulator& destination) const {
        destination = source;
    }

private:
    template<typename Sum>
    int64_t outputScalar(const int32_t* firstAccumulator, const int32_t* secondAccumulator) const {
        Sum output = outputBias_;
        for(size_t hidden = 0; hidden < hiddenSize_; hidden++){
            const int32_t first = std::clamp(firstAccumulator[hidden], 0, hiddenScale_);
            const int32_t second = std::clamp(secondAccumulator[hidden], 0, hiddenScale_);
            output += static_cast<Sum>(first) * outputWeights_[hidden];
            output += static_cast<Sum>(second) * outputWeights_[hiddenSize_ + hidden];
        }
        return output;
    }

#if defined(FORKLIFT_NNUE_HAS_SSE2)
    int64_t outputSse2(const int32_t* firstAccumulator, const int32_t* secondAccumulator) const {
        const __m128i zero = _mm_setzero_si128();
        const __m128i ceiling = _mm_set1_epi16(static_cast<int16_t>(hiddenScale_));
        __m128i sum = zero;
        size_t hidden = 0;
        for(; hidden + 8 <= hiddenSize_; hidden += 8){
            const auto clipped = [&](const int32_t* values){
                const __m128i low = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values));
                const __m128i high = _mm_loadu_si128(reinterpret_cast<const __m128i*>(values + 4));
                // Saturating to signed int16 before clipping is exact because
                // 0 <= hiddenScale <= INT16_MAX. Even INT32_MIN/MAX inputs
                // therefore produce the same activation as the scalar path.
                return _mm_min_epi16(_mm_max_epi16(_mm_packs_epi32(low, high), zero), ceiling);
            };
            const __m128i first = clipped(firstAccumulator + hidden);
            const __m128i second = clipped(secondAccumulator + hidden);
            const __m128i firstWeights = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(outputWeights_.data() + hidden));
            const __m128i secondWeights = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(outputWeights_.data() + hiddenSize_ + hidden));
            sum = _mm_add_epi32(sum, _mm_madd_epi16(first, firstWeights));
            sum = _mm_add_epi32(sum, _mm_madd_epi16(second, secondWeights));
        }
        // Each lane and each horizontal partial sum is covered by the same
        // absolute-bound certificate. Unaligned loads and the scalar tail
        // support arbitrary format-valid hidden sizes without padding.
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        int32_t output = _mm_cvtsi128_si32(sum) + outputBias_;
        for(; hidden < hiddenSize_; hidden++){
            const int32_t first = std::clamp(firstAccumulator[hidden], 0, hiddenScale_);
            const int32_t second = std::clamp(secondAccumulator[hidden], 0, hiddenScale_);
            output += first * outputWeights_[hidden];
            output += second * outputWeights_[hiddenSize_ + hidden];
        }
        return output;
    }
#endif

    template<typename T>
    static bool readValue(std::ifstream& input, T& value){
        input.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
        return static_cast<bool>(input);
    }

    template<typename T>
    static bool readArray(std::ifstream& input, std::vector<T>& values){
        if(values.empty()) return true;
        input.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(T)));
        return static_cast<bool>(input);
    }

    static int pieceBucket(PieceType type){
        switch(type){
            case PieceType::Pawn: return 0;
            case PieceType::Knight: return 1;
            case PieceType::Bishop: return 2;
            case PieceType::Rook: return 3;
            case PieceType::Queen: return 4;
            default: return -1;
        }
    }

    static int orientSquare(int square, Color perspective){
        if(perspective == Color::White) return square;
        const int file = square % 8;
        const int rank = 7 - square / 8;
        return rank * 8 + file;
    }

    static size_t featureIndex(const Board& board, Color perspective, int pieceSquare, const Piece& piece){
        const int king = orientSquare(board.findKing(perspective), perspective);
        const int relativeColor = piece.c == perspective ? 0 : 1;
        const int bucket = relativeColor * 5 + pieceBucket(piece.t);
        const int square = orientSquare(pieceSquare, perspective);
        return static_cast<size_t>(king * 640 + bucket * 64 + square);
    }

    static size_t colorIndex(Color color){
        return color == Color::White ? 0U : 1U;
    }

    const int16_t* featureWeights(const Board& board, Color perspective, int square,
                                  const Piece& piece) const {
        return inputWeights_.data() + featureIndex(board, perspective, square, piece) * hiddenSize_;
    }

    void addFeature(const Board& board, Color perspective, int square, const Piece& piece,
                    int sign, std::vector<int32_t>& accumulator) const {
        const int16_t* weights = featureWeights(board, perspective, square, piece);
        for(size_t hidden = 0; hidden < hiddenSize_; hidden++){
            accumulator[hidden] += sign * weights[hidden];
        }
    }

    void refreshPerspective(const Board& board, Color perspective,
                            std::vector<int32_t>& accumulator) const {
        accumulator = hiddenBias_;
        addActiveFeatures(board, perspective, accumulator);
    }

    void addActiveFeatures(const Board& board, Color perspective, std::vector<int32_t>& accumulator) const {
        for(int square = 0; square < 64; square++){
            const Piece piece = board.b[static_cast<size_t>(square)];
            if(isNone(piece) || piece.t == PieceType::King) continue;
            const int bucket = pieceBucket(piece.t);
            if(bucket < 0) continue;
            addFeature(board, perspective, square, piece, 1, accumulator);
        }
    }

    u32 hiddenSize_ = 0;
    int32_t hiddenScale_ = 0;
    int32_t outputScale_ = 0;
    int32_t outputBias_ = 0;
    bool boundedOutput32_ = false;
    std::vector<int32_t> hiddenBias_;
    std::vector<int16_t> inputWeights_;
    std::vector<int16_t> outputWeights_;
    std::string source_;
};

#if defined(FORKLIFT_NNUE_HAS_SSE2)
#undef FORKLIFT_NNUE_HAS_SSE2
#endif
