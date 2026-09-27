#pragma once

#include "board.hpp"

struct TimeBudget {
    int softMs = 1000;
    int hardMs = 1000;
};

enum class GuiTimeProfile {
    Eco,
    Balanced,
    Performance,
    PerformancePlus
};

struct GuiTimeSuggestion {
    int limitMs = 1000;
    int profileCapMs = 1000;
    int complexityPercent = 100;
    int weightPercent = 100;
};

inline const char* guiTimeProfileName(GuiTimeProfile profile){
    switch(profile){
        case GuiTimeProfile::Eco: return "eco";
        case GuiTimeProfile::Balanced: return "balanced";
        case GuiTimeProfile::Performance: return "performance";
        case GuiTimeProfile::PerformancePlus: return "performance+";
    }
    return "balanced";
}

inline int guiTimeProfileWeightPercent(GuiTimeProfile profile){
    switch(profile){
        case GuiTimeProfile::Eco: return 55;
        case GuiTimeProfile::Balanced: return 85;
        case GuiTimeProfile::Performance: return 130;
        case GuiTimeProfile::PerformancePlus: return 180;
    }
    return 85;
}

inline int guiTimeProfileCapPercent(GuiTimeProfile profile){
    switch(profile){
        case GuiTimeProfile::Eco: return 10;
        case GuiTimeProfile::Balanced: return 25;
        case GuiTimeProfile::Performance: return 50;
        case GuiTimeProfile::PerformancePlus: return 100;
    }
    return 25;
}

inline int guiTimeProfileCapMs(GuiTimeProfile profile, int autoMaxMs){
    const int maximum = std::clamp(autoMaxMs, 1000, 300000);
    const long long scaled = static_cast<long long>(maximum) * guiTimeProfileCapPercent(profile) / 100;
    return std::clamp(static_cast<int>(scaled), 1000, maximum);
}

inline int countNonKingPieces(const Board& board){
    int pieces = 0;
    for(const Piece& p : board.b){
        if(isNone(p) || p.t == PieceType::King) continue;
        pieces++;
    }
    return pieces;
}

inline int positionComplexityScale(const Board& board, int legalMoves){
    if(legalMoves <= 1) return 35;

    int scale = 100;
    const int pieces = countNonKingPieces(board);
    if(board.inCheck(board.stm)) scale += 20;

    if(legalMoves >= 36) scale += 18;
    else if(legalMoves >= 28) scale += 10;
    else if(legalMoves <= 6) scale -= 18;
    else if(legalMoves <= 10) scale -= 10;

    if(pieces >= 20) scale += 8;
    else if(pieces <= 8) scale -= 8;

    return std::clamp(scale, 35, 145);
}

inline TimeBudget pickClockTimeBudget(const Board& board,
                                      int sideTime,
                                      int sideIncrement,
                                      int movesToGo,
                                      int moveOverheadMs){
    constexpr int MaximumClockComponentMs = 24 * 60 * 60 * 1000;
    const int increment = std::clamp(sideIncrement, 0, MaximumClockComponentMs);
    Board probe = board;
    MoveList legal;
    probe.genLegalMoves(legal);
    const int legalMoves = static_cast<int>(legal.size());
    const int complexityScale = positionComplexityScale(board, legalMoves);

    if(movesToGo <= 0){
        const int pieces = countNonKingPieces(board);
        movesToGo = pieces >= 22 ? 32 : (pieces >= 12 ? 24 : 16);
    }

    // Preserve enough wall-clock margin for UCI transport and process
    // scheduling. This matters when the remaining clock is roughly one
    // increment: spending the entire increment can still lose on time after
    // the bestmove line leaves the engine process.
    const int reserve = std::max(std::clamp(moveOverheadMs, 0, 5000),
                                 std::min(250, std::max(1, sideTime / 50)));
    const int safeTime = std::max(1, sideTime - reserve);
    const int baseSlice = safeTime / std::max(1, movesToGo + 3);
    int soft = baseSlice + (increment * 3) / 4;
    if(movesToGo <= 8) soft += baseSlice / 3;
    if(sideTime < 2000) soft = std::max(5, baseSlice + increment / 2);
    soft = (soft * complexityScale) / 100;
    if(legalMoves <= 1) soft = std::min(soft, std::max(5, std::min(80, safeTime / 20)));
    soft = std::clamp(soft, 1, std::max(1, safeTime / 2));

    int hard = std::max(soft + 40, soft + soft / 2);
    hard = std::max(hard, baseSlice * 3 + increment);
    if(sideTime < 1000) hard = std::max(soft + 20, soft * 2);
    hard = (hard * std::max(100, complexityScale + 10)) / 100;
    if(legalMoves <= 1) hard = std::min(hard, std::max(soft, std::min(120, safeTime / 12)));
    hard = std::clamp(hard, soft, safeTime);

    // When less than roughly two increments remain, repeatedly spending the
    // full increment is not stable: scheduling and UCI transport overhead can
    // make each nominally level move leak clock until a forfeit. Enter a
    // recovery regime that searches well below the increment so the clock can
    // rebuild. Sudden-death controls keep the normal allocation above.
    const long long recoveryThreshold = static_cast<long long>(increment) * 2 + reserve;
    if(increment > 0 && sideTime <= recoveryThreshold){
        const int recoveryHard = std::max(
            1, std::min(safeTime / 3, std::max(1, increment / 2)));
        hard = std::min(hard, recoveryHard);
        soft = std::min(soft, std::max(1, hard / 2));
        hard = std::max(hard, soft);
    }
    return TimeBudget{soft, hard};
}

inline GuiTimeSuggestion suggestGuiTimeLimit(const Board& board,
                                             int requestedMs,
                                             int autoMaxMs,
                                             GuiTimeProfile profile){
    Board probe = board;
    std::vector<Move> legal;
    probe.genLegalMoves(legal);
    const int legalMoves = static_cast<int>(legal.size());
    const int complexityScale = positionComplexityScale(board, legalMoves);

    const int base = std::clamp(requestedMs, 100, 300000);
    const int weight = guiTimeProfileWeightPercent(profile);
    const int profileCap = guiTimeProfileCapMs(profile, autoMaxMs);
    const long long weighted = static_cast<long long>(base) * complexityScale * weight / 10000;
    int suggestion = std::clamp(static_cast<int>(weighted), 100, profileCap);
    if(legalMoves <= 1) suggestion = std::min(suggestion, std::max(100, base / 5));
    return GuiTimeSuggestion{suggestion, profileCap, complexityScale, weight};
}

inline TimeBudget pickGuiTimeBudget(const Board&, int requestedMs, int hardCapMs = 300000){
    const int cap = std::clamp(hardCapMs, 100, 300000);
    const int soft = std::clamp(requestedMs, 100, cap);
    const int hard = std::clamp(soft + std::max(250, soft / 3), soft, cap);
    return TimeBudget{soft, hard};
}
