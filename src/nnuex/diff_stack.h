/*
  Minimal piece-diff stack for NNUEX search integration.
*/

#ifndef NNUEX_DIFF_STACK_H_INCLUDED
#define NNUEX_DIFF_STACK_H_INCLUDED

#include <array>
#include <cassert>
#include <cstddef>

#include "../types.h"

namespace Stockfish::Eval::NNUEX {

template<std::size_t MaxSize = std::size_t(MAX_PLY) + 1>
class PieceDiffStack {
   public:
    void reset() noexcept { size_ = 1; }

    DirtyPiece& push() noexcept {
        assert(size_ < MaxSize);
        auto& dirtyPiece = dirtyPieces_[size_];
        ++size_;
        return dirtyPiece;
    }

    void pop() noexcept {
        assert(size_ > 1);
        --size_;
    }

   private:
    std::array<DirtyPiece, MaxSize> dirtyPieces_;
    std::size_t                     size_ = 1;
};

}  // namespace Stockfish::Eval::NNUEX

#endif  // NNUEX_DIFF_STACK_H_INCLUDED
