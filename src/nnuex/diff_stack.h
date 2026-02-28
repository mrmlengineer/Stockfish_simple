/*
  Minimal move-diff stack for NNUEX search integration.
*/

#ifndef NNUEX_DIFF_STACK_H_INCLUDED
#define NNUEX_DIFF_STACK_H_INCLUDED

#include <array>
#include <cassert>
#include <cstddef>
#include <new>
#include <utility>

#include "../types.h"

namespace Stockfish::Eval::NNUEX {

template<std::size_t MaxSize = std::size_t(MAX_PLY) + 1>
class DiffStack {
   public:
    void reset() noexcept { size_ = 1; }

    std::pair<DirtyPiece&, DirtyThreats&> push() noexcept {
        assert(size_ < MaxSize);
        auto& dirtyPiece   = dirtyPieces_[size_];
        auto& dirtyThreats = dirtyThreats_[size_];
        new (&dirtyThreats) DirtyThreats;
        ++size_;
        return {dirtyPiece, dirtyThreats};
    }

    void pop() noexcept {
        assert(size_ > 1);
        --size_;
    }

   private:
    std::array<DirtyPiece, MaxSize>   dirtyPieces_;
    std::array<DirtyThreats, MaxSize> dirtyThreats_;
    std::size_t                       size_ = 1;
};

}  // namespace Stockfish::Eval::NNUEX

#endif  // NNUEX_DIFF_STACK_H_INCLUDED
