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
  /// Returns an empty Optional if the value has no uses.
  llvm::Optional<size_t> getFirstUseIndex(Value value) {
    auto intervals = useIntervalMap[value];
    return intervals.empty() ? llvm::None
                             : llvm::Optional<size_t>(intervals.begin()->first);
  }

  /// Checks if the use intervals of the given values interfere.
  bool rangesInterfere(Value itemA, Value itemB) const;

  /// Merges the userange of itemB into the userange of itemA.
  /// Note: This assumes that there is no interference between the two
  /// ranges.
  void unionRanges(Value itemA, Value itemB);

  /// Dumps the liveness information to the given stream.
  void print(raw_ostream &os);

private:
  using ValueSetT = BufferAliasAnalysis::ValueSetT;
  using OperationListT = Liveness::OperationListT;

  /// Builds an IntervalVector corresponding to the given OperationList.
  IntervalVector computeInterval(const Liveness::OperationListT &operationList);

  /// Merge two sorted (by operationID) OperationLists and ignore double
  /// entries. Return the new computed OperationList.
  Liveness::OperationListT
  mergeUseranges(const Liveness::OperationListT &first,
                 const Liveness::OperationListT &second) const;

  /// Performs an interval union of the interval vectors from the given values.
  /// Returns an empty Optional if there is an interval interference.
  llvm::Optional<IntervalVector> intervalUnion(Value itemA, Value itemB) const;

  /// Performs an interval subtraction => A = A - B.
  /// Note: This assumes that all intervals of b are included in some interval
  ///       of a.
  void intervalSubtract(IntervalVector &a, const IntervalVector &b) const;

  /// Maps each Operation to a unique ID according to the program squence.
  DenseMap<Operation *, size_t> operationIds;

  /// Maps a value to its use range interval.
  DenseMap<Value, IntervalVector> useIntervalMap;

  /// Cache the alias lists for all values to avoid recomputation.
  BufferAliasAnalysis::ValueMapT aliasCache;

  /// The current liveness info.
  Liveness liveness;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_USERANGE_H
