//===- UserangeAnalysis.h - Userange analysis for MLIR ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains an analysis for computing the userange information for a
// given value. This version uses the liverange information to compute a vector
// of integer intervals representing the userange.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_USERANGE_H
#define MLIR_ANALYSIS_USERANGE_H

#include <vector>

#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/BufferUtils.h"

namespace mlir {

/// Represents an analysis for computing the useranges of all alloc values
/// inside a given operation. The analysis uses liveness information to compute
/// intervals starting at the first and ending with the last use of every alloc
/// value.
class UserangeAnalysis {

public:
  using UseInterval = std::pair<size_t, size_t>;
  using IntervalVector = SmallVector<UseInterval, 8>;

  UserangeAnalysis(Operation *op, BufferPlacementAllocs allocs,
                   BufferAliasAnalysis aliases);

  /// Returns the index of the first operation that uses the given value.
  size_t getFirstUseIndex(Value value) {
    return !useIntervalMap[value].empty() ? useIntervalMap[value].begin()->first
                                          : -1;
  }

  /// Check if a reuse of two values and their first and last uses is possible.
  /// It depends on userange interferences, alias interference and real uses.
  /// Returns true if a reuse is possible.
  bool rangeInterference(Value itemA, Value itemB);

  /// Merges the userange of itemB into the userange of itemA.
  /// Note: This assumes that there is no interference between the two
  /// ranges.
  void rangeUnion(Value itemA, Value itemB);

  /// Dumps the liveness information to the given stream.
  void print(raw_ostream &os);

private:
  using OperationListT = Liveness::OperationListT;
  using ValueSetT = BufferAliasAnalysis::ValueSetT;

  /// Computes the userange of the given value by iterating over all of its
  /// uses.
  OperationListT computeUserange(Value value);

  IntervalVector computeInterval(Value value, OperationListT &operationList);

  /// This performs an interval union of two sorted interval vectors.
  /// Return false if there is an interval interference.
  std::pair<IntervalVector, bool> intervalUnion(IntervalVector intervalA,
                                                IntervalVector intervalB) const;

  /// Removes all values that are in b from a.
  /// Note: This assumes that all intervals of b are included in some interval
  ///       of a.
  void intervalSubtract(IntervalVector &a, IntervalVector &b) const;

  /// Merge two sorted OperationLists into the first OperationList and
  /// ignores double entries.
  void mergeUseranges(OperationListT &first, OperationListT &second);

  /// Adds all operations from start to end to the OperationList including
  /// nested regions to the userange of the given value. If includeEnd is false
  /// the end operation is not added.
  template <bool includeEnd = true>
  void addAllOperationsBetween(Value value, Operation *start, Operation *end) {
    if (includeEnd) {
      result.push_back(start);
      addAllOperationsInRegion(value, start);
    }

    while (start != end) {
      if (includeEnd)
        start = start->getNextNode();
      addAllOperationsInRegion(value, start);
      result.push_back(start);
      if (!includeEnd)
        start = start->getNextNode();
    }
  }

  /// Iterates over all regions of a given operation and adds all operations
  /// inside those regions to the userange of the given value.
  void addAllOperationsInRegion(Value value, Operation *parentOp);

  /// Find the starting operation of the given value inside the given block.
  Operation *getStartOperation(Value value, Block *block);

  /// Adds the correct operations in the given block, and potentially its
  /// successors, to the userange of the given value. The startBlock is the
  /// block at which the successor chain started and is used as an anchor if a
  /// loop is found.
  void processSuccessor(Value value, Block *block, Block *startBlock);

  /// Finds the top level block that has the given value in its liveOut set.
  Block *findTopLiveBlock(Value value, Operation *op);

  /// Finds the top level parentOp that implements a LoopLikeOpInterface.
  /// Returns nullptr if none exists.
  Operation *findParentLoopOp(Operation *op);

  /// The result list of the userange computation.
  OperationListT result;

  /// The list of visited blocks during the userange computation.
  SmallPtrSet<Block *, 32> visited;

  /// The list of blocks that the userange computation started from.
  SmallPtrSet<Block *, 8> startBlocks;

  /// Maps each Operation to an ID.
  DenseMap<Operation *, size_t> operationIds;

  /// Maps a value to their use range interval.
  DenseMap<Value, IntervalVector> useIntervalMap;

  /// Cache the alias lists for all values to avoid the recomputation.
  BufferAliasAnalysis::ValueMapT aliasCache;

  /// The current liveness info.
  Liveness liveness;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_USERANGE_H