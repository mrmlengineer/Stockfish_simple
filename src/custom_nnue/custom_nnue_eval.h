/*
  Custom NNUEX outer evaluation integration (full mode).
*/

#ifndef CUSTOM_NNUE_EVAL_H_INCLUDED
#define CUSTOM_NNUE_EVAL_H_INCLUDED

#include <string>

#include "../types.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::CustomNNUE {

class Network;
struct IncrementalState;

Value       evaluate(const Network& net, const Position& pos, int optimism, bool incrementalRequested);
Value       evaluate(const Network&         net,
                     const Position&        pos,
                     const IncrementalState* incrementalState,
                     int                    optimism,
                     bool                   incrementalRequested);
std::string trace(Position& pos, const Network& net);

}  // namespace Stockfish::CustomNNUE

#endif  // CUSTOM_NNUE_EVAL_H_INCLUDED
