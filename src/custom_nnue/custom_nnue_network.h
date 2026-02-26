/*
  Custom dual-head NNUEX runtime (full-recompute reference path).
*/

#ifndef CUSTOM_NNUE_NETWORK_H_INCLUDED
#define CUSTOM_NNUE_NETWORK_H_INCLUDED

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "../types.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::CustomNNUE {

inline constexpr const char* EvalFileDefaultName =
  "quantized_weights_qat_only3_split_h1_dual_heads_stm16buckets.nnuex";

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

    // Integer runtime domains (matching Python integer-sim semantics):
    // - h1Pre: q255 pre-activation accumulator
    // - h1Clip: q127 clipped activation
    std::array<std::int32_t, 512> h1Pre{};
    std::array<std::uint8_t, 512> h1Clip{};
};

struct RuntimeMetrics {
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

class ScopedRuntimeMetricsBinding {
   public:
    explicit ScopedRuntimeMetricsBinding(RuntimeMetrics* sink) noexcept;
    ~ScopedRuntimeMetricsBinding() noexcept;

    ScopedRuntimeMetricsBinding(const ScopedRuntimeMetricsBinding&)            = delete;
    ScopedRuntimeMetricsBinding& operator=(const ScopedRuntimeMetricsBinding&) = delete;

   private:
    RuntimeMetrics* prev_ = nullptr;
};

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
    bool                      advance_incremental_state(const Position&       posAfterMove,
                                                        Move                  move,
                                                        const DirtyPiece&     dirtyPiece,
                                                        const IncrementalState& prev,
                                                        IncrementalState&     next) const;
    bool                      advance_incremental_state_null(const Position&        posAfterNull,
                                                             const IncrementalState& prev,
                                                             IncrementalState&      next) const;

   private:
    std::string resolve_evalfile_path(const std::string& rootDirectory,
                                     const std::string& requestedPath) const;
    bool        load_from_file(const std::string& path, std::string& err);

    bool        initialized_ = false;
    std::string requestedPath_;
    std::string loadedPath_;
    std::string error_;
    Impl*       impl_ = nullptr;

   public:
    // Custom copy/move to keep pimpl owned without exposing headers in shm/numa templates.
    Network(const Network&);
    Network(Network&&) noexcept;
    Network& operator=(const Network&);
    Network& operator=(Network&&) noexcept;
    ~Network();
};

}  // namespace Stockfish::CustomNNUE

#endif  // CUSTOM_NNUE_NETWORK_H_INCLUDED
