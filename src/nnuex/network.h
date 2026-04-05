/*
  NNUEX runtime (full-recompute reference path).
*/

#ifndef NNUEX_NETWORK_H_INCLUDED
#define NNUEX_NETWORK_H_INCLUDED

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#ifndef NNUEX_H1_POSITIONAL
  #define NNUEX_H1_POSITIONAL 384
#endif

#include "../types.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUEX {

#define EvalFileDefaultNameNNUEX \
  "qat_b16_exact_2500_alt_psqt50_dual_positional_pairreg_i16ft256_positional_i16_ft256_psqt256_quantized_weights.nnuex"

inline constexpr const char* EvalFileDefaultName = EvalFileDefaultNameNNUEX;

struct Evaluation {
    Value psqt       = VALUE_ZERO;
    Value positional = VALUE_ZERO;
    int   pieceCount = 0;
    int   bucket8    = 0;
    int   stmBlack   = 0;
    int   bucket16   = 0;
    float psqtNorm   = 0.0f;
    float posNorm    = 0.0f;
};

struct IncrementalState {
    bool valid = false;
    Key  key   = 0;

    int pieceCount = 0;
    int bucket8    = 0;
    int stmBlack   = 0;
    int bucket16   = 0;

    // New-architecture incremental domains:
    // - positionalH1Pre: i16 pre-activation accumulator for positional_hidden_1
    // - positionalH1Clip: q127 clipped activation for positional_hidden_1
    // - psqtBucketAcc: unsigned direct PSQT accumulator for all 8 piece buckets
    alignas(64) std::array<std::int16_t, NNUEX_H1_POSITIONAL> positionalH1Pre{};
    alignas(64) std::array<std::uint8_t, NNUEX_H1_POSITIONAL> positionalH1Clip{};
    alignas(32) std::array<std::int32_t, 8>                   psqtBucketAcc{};
};

struct RuntimeMetrics {
    std::uint64_t encodePositionCalls      = 0;
    std::uint64_t encodePositionNs         = 0;
    std::uint64_t h1ClipCalls            = 0;
    std::uint64_t h1ClipNs               = 0;
    std::uint64_t postH1ForwardCalls     = 0;
    std::uint64_t postH1ForwardNs        = 0;
    std::uint64_t outputConvertCalls     = 0;
    std::uint64_t outputConvertNs        = 0;
    std::uint64_t buildPreClipCalls      = 0;
    std::uint64_t buildPreClipNs         = 0;
    std::uint64_t advanceMovePreClipCalls = 0;
    std::uint64_t advanceMovePreClipNs    = 0;
    std::uint64_t advanceNullPreClipCalls = 0;
    std::uint64_t advanceNullPreClipNs    = 0;
};

#ifndef NNUEX_FIXED_MODE
class ScopedRuntimeMetricsBinding {
   public:
    explicit ScopedRuntimeMetricsBinding(RuntimeMetrics* sink) noexcept;
    ~ScopedRuntimeMetricsBinding() noexcept;

    ScopedRuntimeMetricsBinding(const ScopedRuntimeMetricsBinding&)            = delete;
    ScopedRuntimeMetricsBinding& operator=(const ScopedRuntimeMetricsBinding&) = delete;

   private:
    RuntimeMetrics* prev_ = nullptr;
};
#else
class ScopedRuntimeMetricsBinding {
   public:
    explicit ScopedRuntimeMetricsBinding(RuntimeMetrics*) noexcept {}
    ~ScopedRuntimeMetricsBinding() noexcept {}
    ScopedRuntimeMetricsBinding(const ScopedRuntimeMetricsBinding&)            = delete;
    ScopedRuntimeMetricsBinding& operator=(const ScopedRuntimeMetricsBinding&) = delete;
};
#endif

class Network {
   public:
    struct Impl;

    Network() = default;

    void load(const std::string& rootDirectory, std::string evalfilePath);
    void verify(std::string evalfilePath, const std::function<void(std::string_view)>& out) const;

    bool               is_initialized() const { return initialized_; }
    const std::string& loaded_path() const { return loadedPath_; }
    const std::string& last_error() const { return error_; }

    std::optional<Evaluation> evaluate(const Position& pos) const;
    std::optional<Evaluation> evaluate(const IncrementalState& state) const;
    bool                      build_incremental_state(const Position& pos, IncrementalState& out) const;
    bool                      advance_incremental_state_from_meta(Move                  move,
                                                                  const DirtyPiece&     dirtyPiece,
                                                                  const IncrementalState& prev,
                                                                  Key                   nextKey,
                                                                  int                   nextStmBlack,
                                                                  IncrementalState&     next) const;
    bool                      advance_incremental_state_null_from_meta(const IncrementalState& prev,
                                                                       Key                   nextKey,
                                                                       int                   nextStmBlack,
                                                                       IncrementalState&     next) const;

   private:
    std::string resolve_evalfile_path(const std::string& rootDirectory,
                                     const std::string& requestedPath) const;
    bool        load_from_file(const std::string& path, std::string& err);
    bool        load_from_memory(const unsigned char* data, std::size_t size, std::string& err);

    bool        initialized_ = false;
    std::string requestedPath_;
    std::string loadedPath_;
    std::string error_;
    Impl*       impl_ = nullptr;

   public:
    // Keep pimpl ownership local without exposing implementation details in numa templates.
    Network(const Network&);
    Network(Network&&) noexcept;
    Network& operator=(const Network&);
    Network& operator=(Network&&) noexcept;
    ~Network();
};

}  // namespace Stockfish::Eval::NNUEX

#endif  // NNUEX_NETWORK_H_INCLUDED
