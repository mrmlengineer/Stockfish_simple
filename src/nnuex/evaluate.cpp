/*
  NNUEX outer evaluation integration.
*/

#include "evaluate.h"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "../evaluate.h"
#include "../position.h"
#include "../uci.h"
#include "network.h"

namespace Stockfish::Eval::NNUEX {

Value evaluate(const Network& net, const Position& pos, int optimism, bool incrementalRequested) {
    return evaluate(net, pos, nullptr, optimism, incrementalRequested);
}

Value evaluate(const Network&          net,
               const Position&         pos,
               const IncrementalState* incrementalState,
               int                     optimism,
               bool                    /*incrementalRequested*/) {
    auto out = incrementalState ? net.evaluate(*incrementalState) : net.evaluate(pos);
    if (!out.has_value())
        return detail::fallback_simple_eval(pos);
    return detail::apply_outer_scaling(pos, *out, optimism);
}

std::string trace(Position& pos, const Network& net) {
    std::stringstream ss;
    if (pos.checkers())
        return "Final evaluation: none (in check)";

    auto out = net.evaluate(pos);
    if (!out.has_value())
    {
        ss << "NNUEX: not loaded\n";
        ss << "Fallback simple_eval: " << Eval::simple_eval(pos);
        return ss.str();
    }

    Value psqt       = out->psqt;
    Value positional = out->positional;
    Value nnue       = (125 * psqt + 131 * positional) / 128;
    Value final      = evaluate(net, pos, VALUE_ZERO, false);

    Value psqtW  = pos.side_to_move() == WHITE ? psqt : -psqt;
    Value posW   = pos.side_to_move() == WHITE ? positional : -positional;
    Value nnueW  = pos.side_to_move() == WHITE ? nnue : -nnue;
    Value finalW = pos.side_to_move() == WHITE ? final : -final;

    ss << "NNUEX trace\n";
    ss << "  piece_count: " << out->pieceCount << "\n";
    ss << "  bucket8/stm/bucket16: " << out->bucket8 << "/" << out->stmBlack << "/" << out->bucket16 << "\n";
    ss << "  psqt (cp, stm): " << int(psqt) << "\n";
    ss << "  positional (cp, stm): " << int(positional) << "\n";
    ss << "  nnue mixed (cp, stm): " << int(nnue) << "\n";
    ss << "  psqt (white cp): " << int(psqtW) << "\n";
    ss << "  positional (white cp): " << int(posW) << "\n";
    ss << "  nnue mixed (white cp): " << int(nnueW) << "\n";
    ss << "  final eval (white cp): " << 0.01 * UCIEngine::to_cp(finalW, pos);
    return ss.str();
}

}  // namespace Stockfish::Eval::NNUEX
