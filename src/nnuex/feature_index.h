/*
  Shared fixed board-feature mapping for the active NNUEX geometry.
  Piece-square features occupy indices 0..735; slot 736 is reserved for the
  legacy STM row and is not produced by this mapper.
*/

#ifndef NNUEX_FEATURE_INDEX_H_INCLUDED
#define NNUEX_FEATURE_INDEX_H_INCLUDED

#include <array>
#include <cstdint>

#include "../types.h"

namespace Stockfish::Eval::NNUEX {

inline constexpr std::uint16_t InvalidFeatureIndex = 0xFFFFu;

struct SquareFeatureGeom {
    int base   = 0;
    int nTypes = 0;
};

inline const std::array<SquareFeatureGeom, SQUARE_NB>& square_feature_geoms() {
    static const std::array<SquareFeatureGeom, SQUARE_NB> geoms = [] {
        std::array<SquareFeatureGeom, SQUARE_NB> out{};
        int                                      offset = 0;
        for (int rank = 8; rank >= 1; --rank)
        {
            const bool edge          = rank == 1 || rank == 8;
            const int  nTypes        = edge ? 5 : 6;
            const int  featuresPerSq = nTypes * 2;
            for (int file = 0; file < 8; ++file)
            {
                const Square sq = make_square(File(file), Rank(rank - 1));
                out[std::size_t(sq)] = {offset + file * featuresPerSq, nTypes};
            }
            offset += 8 * featuresPerSq;
        }
        return out;
    }();
    return geoms;
}

constexpr int piece_index_no_pawn(PieceType pt) {
    switch (pt)
    {
    case ROOK:
        return 0;
    case KNIGHT:
        return 1;
    case BISHOP:
        return 2;
    case QUEEN:
        return 3;
    case KING:
        return 4;
    default:
        return -1;
    }
}

constexpr int piece_index_all(PieceType pt) {
    switch (pt)
    {
    case PAWN:
        return 0;
    case ROOK:
        return 1;
    case KNIGHT:
        return 2;
    case BISHOP:
        return 3;
    case QUEEN:
        return 4;
    case KING:
        return 5;
    default:
        return -1;
    }
}

constexpr std::size_t piece_square_feature_slot(Piece pc, Square sq) {
    return std::size_t(pc) * SQUARE_NB + std::size_t(sq);
}

inline const std::array<std::uint16_t, PIECE_NB * SQUARE_NB>& piece_square_feature_indices() {
    constexpr int kBoardFeatureCount = 736;
    static const std::array<std::uint16_t, PIECE_NB * SQUARE_NB> indices = [] {
        std::array<std::uint16_t, PIECE_NB * SQUARE_NB> out{};
        for (auto& featureIndex : out)
            featureIndex = InvalidFeatureIndex;

        for (int pcValue = 0; pcValue < PIECE_NB; ++pcValue)
        {
            const Piece pc = Piece(pcValue);
            if (pc == NO_PIECE)
                continue;

            for (int sqValue = 0; sqValue < SQUARE_NB; ++sqValue)
            {
                const Square sq   = Square(sqValue);
                const auto&  geom = square_feature_geoms()[std::size_t(sq)];
                const int    idx = geom.nTypes == 5 ? piece_index_no_pawn(type_of(pc))
                                                    : piece_index_all(type_of(pc));
                if (idx < 0)
                    continue;

                const int colorOffset = color_of(pc) == BLACK ? geom.nTypes : 0;
                const int outIndex    = geom.base + colorOffset + idx;
                if (outIndex < 0 || outIndex >= kBoardFeatureCount)
                    continue;

                out[piece_square_feature_slot(pc, sq)] = static_cast<std::uint16_t>(outIndex);
            }
        }

        return out;
    }();
    return indices;
}

inline std::uint16_t feature_index_for_piece_square(Piece pc, Square sq) noexcept {
    if (pc == NO_PIECE || sq == SQ_NONE || !is_ok(sq))
        return InvalidFeatureIndex;
    return piece_square_feature_indices()[piece_square_feature_slot(pc, sq)];
}

inline bool feature_index_for_piece_square(Piece pc, Square sq, std::uint16_t& featureIndex) {
    featureIndex = feature_index_for_piece_square(pc, sq);
    return featureIndex != InvalidFeatureIndex;
}

inline bool feature_index_for_piece_square(Piece pc, Square sq, int& featureIndex) {
    std::uint16_t out = InvalidFeatureIndex;
    if (!feature_index_for_piece_square(pc, sq, out))
        return false;
    featureIndex = int(out);
    return true;
}

}  // namespace Stockfish::Eval::NNUEX

#endif  // NNUEX_FEATURE_INDEX_H_INCLUDED
