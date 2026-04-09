/*
  NNUEX outer evaluation integration.
*/

#ifndef NNUEX_EVALUATE_H_INCLUDED
#define NNUEX_EVALUATE_H_INCLUDED

#include <algorithm>
#include <cmath>
#include <string>

#include "../evaluate.h"
#include "../position.h"
#include "../types.h"
#include "network.h"

namespace Stockfish::Eval::NNUEX {

namespace detail {

inline Value fallback_simple_eval(const Position& pos) {
    return std::clamp(Value(Eval::simple_eval(pos)), VALUE_TB_LOSS_IN_MAX_PLY + 1,
                      VALUE_TB_WIN_IN_MAX_PLY - 1);
}

inline Value apply_outer_scaling(const Position& pos, const Evaluation& e, int optimism) {
    Value psqt       = e.psqt;
    Value positional = e.positional;
    Value nnue       = (125 * psqt + 131 * positional) / 128;

    int nnueComplexity = std::abs(int(psqt) - int(positional));
    optimism += optimism * nnueComplexity / 476;
    nnue -= nnue * nnueComplexity / 18236;

    int material = 534 * pos.count<PAWN>() + pos.non_pawn_material();
    int v        = (nnue * (77871 + material) + optimism * (7191 + material)) / 77871;

    v -= v * pos.rule50_count() / 199;
    v = std::clamp(v, VALUE_TB_LOSS_IN_MAX_PLY + 1, VALUE_TB_WIN_IN_MAX_PLY - 1);
    return Value(v);
}

}  // namespace detail

Value       evaluate(const Network& net, const Position& pos, int optimism);
Value       evaluate(const Network&         net,
                     const Position&        pos,
                     const IncrementalState* incrementalState,
                     int                    optimism);

std::string trace(Position& pos, const Network& net);

}  // namespace Stockfish::Eval::NNUEX

#endif  // NNUEX_EVALUATE_H_INCLUDED
