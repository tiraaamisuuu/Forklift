#pragma once

// Diagnostic-only attribution. Production builds contain no counters, timing
// reads or thread-local storage from this header. Timings from an instrumented
// build are not a substitute for paired uninstrumented search benchmarks.
#if defined(FORKLIFT_LEGALGEN_DIAGNOSTICS)
#include <chrono>
#include <cstdint>

struct LegalGenerationDiagnostics {
    std::uint64_t calls = 0;
    std::uint64_t pseudoCandidates = 0;
    std::uint64_t fullMakeUndoProbes = 0;
    std::uint64_t fastAccepted = 0;
    std::uint64_t checkedCalls = 0;
    std::uint64_t elapsedNs = 0;
};

// A diagnostic benchmark uses one search thread. Independent worker threads
// deliberately retain independent counters rather than contending on atomics.
inline thread_local LegalGenerationDiagnostics legalGenerationDiagnostics;

inline void resetLegalGenerationDiagnostics(){
    legalGenerationDiagnostics = {};
}

class LegalGenerationDiagnosticsScope {
public:
    LegalGenerationDiagnosticsScope() : started_(Clock::now()){
        legalGenerationDiagnostics.calls++;
    }

    ~LegalGenerationDiagnosticsScope(){
        legalGenerationDiagnostics.elapsedNs += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started_).count());
    }

    LegalGenerationDiagnosticsScope(const LegalGenerationDiagnosticsScope&) = delete;
    LegalGenerationDiagnosticsScope& operator=(const LegalGenerationDiagnosticsScope&) = delete;

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point started_;
};
#endif
