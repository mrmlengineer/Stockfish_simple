/*
  NNUEX outer evaluation integration.
*/

#ifndef NNUEX_EVALUATE_H_INCLUDED
#define NNUEX_EVALUATE_H_INCLUDED

#include <string>

#include "../types.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUEX {

class Network;
struct IncrementalState;

Value       evaluate(const Network& net, const Position& pos, int optimism, bool incrementalRequested);
Value       evaluate(const Network&         net,
                     const Position&        pos,
                     const IncrementalState* incrementalState,
                     int                    optimism,
                     bool                   incrementalRequested);
std::string trace(Position& pos, const Network& net);

}  // namespace Stockfish::Eval::NNUEX

#endif  // NNUEX_EVALUATE_H_INCLUDED
