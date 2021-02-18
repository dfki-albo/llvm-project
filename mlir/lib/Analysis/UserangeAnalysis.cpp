//===- UserangeAnalysis.cpp - Userange analysis for MLIR -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
// Implementation of the userange analysis.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/UserangeAnalysis.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;

UserangeAnalysis::UserangeAnalysis(Operation *op, BufferPlacementAllocs allocs,
                                   BufferAliasAnalysis aliases)
    : liveness(op) {
  // Walk over all operations and map them to an ID.
  op->walk([&](Operation *operation) {
    operationIds.insert({operation, operationIds.size()});
  });

  // A map for all use ranges of the aliases. This is necessary to prevents
  // double computations of the use range interval of the alias.
  DenseMap<Value, OperationListT> aliasUseranges;
  // Compute the use range for every allocValue and their aliases. Merge them
  // and compute an interval. Add all computed intervals to the
  // useIntervalMap.
  for (BufferPlacementAllocs::AllocEntry entry : allocs) {
    Value allocValue = std::get<0>(entry);
    OperationListT liveOperations = computeUserange(allocValue);
    ValueSetT aliasSet = aliases.resolve(allocValue);
    for (Value alias : aliasSet) {
      if (alias == allocValue)
        continue;
      if (!aliasUseranges.count(alias)) {
        aliasUseranges.insert(std::pair<Value, Liveness::OperationListT>(
            alias, liveness.resolveLiveness(alias)));
        useIntervalMap.insert(std::pair<Value, IntervalVector>(
            alias, computeInterval(alias, aliasUseranges[alias])));
      }
      mergeUseranges(liveOperations, aliasUseranges[alias]);
    }
    aliasCache.insert(std::pair<Value, ValueSetT>(allocValue, aliasSet));
    useIntervalMap.insert(std::pair<Value, IntervalVector>(
        allocValue, computeInterval(allocValue, liveOperations)));
  }
}

/// Check if a reuse of two values and their first and last uses is possible.
/// It depends on userange interferences, alias interference and real uses.
/// Returns true if a reuse is possible.
bool UserangeAnalysis::rangeInterference(Value itemA, Value itemB) {
  ValueSetT intersect = aliasCache[itemA];
  llvm::set_intersect(intersect, aliasCache[itemB]);
  IntervalVector tmpIntervalA = useIntervalMap[itemA];
  IntervalVector tmpIntervalB = useIntervalMap[itemB];

  if (!intersect.empty())
    for (Value alias : intersect) {
      IntervalVector aliasInterval = useIntervalMap[alias];
      intervalSubtract(tmpIntervalA, aliasInterval);
      intervalSubtract(tmpIntervalB, aliasInterval);
    }

  return intervalUnion(tmpIntervalA, tmpIntervalB).second;
}

/// Merges the userange of itemB into the userange of itemA.
/// Note: This assumes that there is no interference between the two
/// ranges.
void UserangeAnalysis::rangeUnion(Value itemA, Value itemB) {
  // Join the aliases of the reusee and reuser.
  llvm::set_union(aliasCache[itemA], aliasCache[itemB]);

  // Compute new interval.
  useIntervalMap[itemA] =
      intervalUnion(useIntervalMap[itemA], useIntervalMap[itemB]).first;
}

/// Computes the use range intervals for the given value and their
/// operationList.
UserangeAnalysis::IntervalVector
UserangeAnalysis::computeInterval(Value value,
                                  Liveness::OperationListT &operationList) {
  assert(!operationList.empty() && "Operation list must not be empty");
  size_t start = operationIds[*operationList.begin()];
  size_t last = start;
  IntervalVector intervals;
  // Iterate over all operations in the operationList. If the gap between the
  // respective operationIds is greater 1 create a new interval.
  for (auto opIter = ++operationList.begin(); opIter != operationList.end();
       ++opIter) {
    size_t current = operationIds[*opIter];
    if (current - last > 1) {
      intervals.push_back(UseInterval(start, last));
      start = current;
    }
    last = current;
  }
  intervals.push_back(UseInterval(start, last));
  return intervals;
}

/// Merge two sorted OperationLists into the first OperationList and ignores
/// double entries.
void UserangeAnalysis::mergeUseranges(Liveness::OperationListT &first,
                                      Liveness::OperationListT &second) {
  Liveness::OperationListT mergeResult;
  auto iterFirst = first.begin();
  auto iterSecond = second.begin();
  // Iterate over the two OperationsLists until one is at the end and insert
  // each operation in order into result.
  for (; iterFirst != first.end() && iterSecond != second.end();) {
    if (operationIds[*iterFirst] < operationIds[*iterSecond])
      mergeResult.push_back(*iterFirst++);
    else if (operationIds[*iterFirst] > operationIds[*iterSecond])
      mergeResult.push_back(*iterSecond++);
    else {
      mergeResult.push_back(*iterFirst++);
      ++iterSecond;
    }
  }
  // Add the remaining Operations to result.
  for (; iterFirst != first.end(); ++iterFirst)
    mergeResult.push_back(*iterFirst);
  for (; iterSecond != second.end(); ++iterSecond)
    mergeResult.push_back(*iterSecond);
  // Overwrite the first OperationList with result.
  first = mergeResult;
}

/// Computes the userange of the given value by iterating over all of its
/// uses.
Liveness::OperationListT UserangeAnalysis::computeUserange(Value value) {
  result.clear();
  visited.clear();
  startBlocks.clear();

  // Iterate over all associated uses
  for (OpOperand &use : value.getUses()) {
    // If one the parents implements a LoopLikeOpInterface we need to add all
    // operations inside of its regions to the userange.
    if (Operation *loopParent = findParentLoopOp(use.getOwner()))
      addAllOperationsInRegion(value, loopParent);

    // Check if the parent block has already been processed.
    Block *useBlock = findTopLiveBlock(value, use.getOwner());
    if (!startBlocks.insert(useBlock).second || visited.contains(useBlock))
      continue;

    // Add all operations inside the block that within the userange of the
    // value.
    const LivenessBlockInfo *blockInfo = liveness.getLiveness(useBlock);
    Operation *start = getStartOperation(value, useBlock);
    Operation *end = blockInfo->getEndOperation(value, start);

    addAllOperationsBetween(value, start, end);

    // If the value is live after the block we need to process the respective
    // successor blocks.
    if (blockInfo->isLiveOut(value)) {
      for (Block *successor : useBlock->getSuccessors()) {
        if (liveness.getLiveness(successor)->isLiveIn(value) &&
            visited.insert(successor).second)
          processSuccessor(value, successor, useBlock);
      }
    }
  }

  // Sort the operation list by the ids.
  std::sort(result.begin(), result.end(),
            [&](Operation *left, Operation *right) {
              return operationIds[left] < operationIds[right];
            });
  return result;
}

/// Finds the top level parentOp that implements a LoopLikeOpInterface.
/// Returns nullptr if none exists.
Operation *UserangeAnalysis::findParentLoopOp(Operation *op) {
  Operation *loopOp = nullptr;
  while (op != nullptr) {
    if (isa<LoopLikeOpInterface>(op))
      loopOp = op;
    op = op->getParentOp();
  }
  return loopOp;
}

/// Finds the top level block that has the given value in its liveOut set.
Block *UserangeAnalysis::findTopLiveBlock(Value value, Operation *op) {
  Operation *topOp = op;
  while (const LivenessBlockInfo *blockInfo =
             liveness.getLiveness(op->getBlock())) {
    if (blockInfo->isLiveOut(value))
      topOp = op;
    op = op->getParentOp();
  }
  return topOp->getBlock();
}

/// Adds the correct operations in the given block, and potentially its
/// successors, to the userange of the given value. The startBlock is the
/// block at which the successor chain started and is used as an anchor if a
/// loop is found.
void UserangeAnalysis::processSuccessor(Value value, Block *block,
                                        Block *startBlock) {
  const LivenessBlockInfo *blockInfo = liveness.getLiveness(block);
  Operation *start = &block->front();
  Operation *end = blockInfo->getEndOperation(value, start);

  addAllOperationsBetween(value, start, end);

  // If the value is live out we need to process all successors at which the
  // value is liveIn.
  if (blockInfo->isLiveOut(value)) {
    for (Block *successor : block->getSuccessors()) {
      // If the successor is the startBlock we have found a loop and only have
      // to add the operations from the block front to the first use of the
      // value.
      if (successor == startBlock) {
        start = &successor->front();
        end = getStartOperation(value, successor);
        addAllOperationsBetween<false>(value, start, end);
        // Else we need to check if the value is liveIn and the successor has
        // not been visited before. If so we also need to process it.
      } else if (liveness.getLiveness(successor)->isLiveIn(value) &&
                 visited.insert(successor).second)
        processSuccessor(value, successor, startBlock);
    }
  }
}

/// Find the starting operation of the given value inside the given block.
Operation *UserangeAnalysis::getStartOperation(Value value, Block *block) {
  Operation *startOperation = &block->back();
  for (Operation *useOp : value.getUsers()) {
    // Find the associated operation in the current block (if any).
    useOp = block->findAncestorOpInBlock(*useOp);
    // Check whether the use is in our block and after the current end
    // operation.
    if (useOp && useOp->isBeforeInBlock(startOperation))
      startOperation = useOp;
  }
  return startOperation;
}

/// Iterates over all regions of a given operation and adds all operations
/// inside those regions to the userange of the given value.
void UserangeAnalysis::addAllOperationsInRegion(Value value,
                                                Operation *parentOp) {
  // Iterate over all regions of the parentOp.
  for (Region &region : parentOp->getRegions()) {
    // Iterate over blocks inside the region.
    for (auto &block : region) {
      // If the blocks has been used as a startBlock before we need to add all
      // operations between the block front and the startOp of the value.
      if (startBlocks.contains(&block)) {
        Operation *start = &block.front();
        Operation *end = getStartOperation(value, &block);
        addAllOperationsBetween<false>(value, start, end);
        // If the block has never been seen we need to add all operations
        // inside it.
      } else if (visited.insert(&block).second) {
        for (Operation &op : block) {
          addAllOperationsInRegion(value, &op);
          result.push_back(&op);
        }
        continue;
      }
      // If the block has either been visited before or was used as a
      // startBlock we need to add all operations between the endOp of the
      // value and the end of the block.
      const LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
      Operation *end = blockInfo->getEndOperation(value, &block.front());
      if (end == &block.back())
        continue;
      addAllOperationsBetween(value, end->getNextNode(), &block.back());
    }
  }
}

/// Removes all values that are in b from a.
/// Note: This assumes that all intervals of b are included in some interval
///       of a.
void UserangeAnalysis::intervalSubtract(IntervalVector &a,
                                        IntervalVector &b) const {
  for (auto iterA = a.begin(), iterB = b.begin();
       iterA != a.end() && iterB != b.end();) {
    // iterA is strictly before iterB => increment iterA
    if (iterA->second < iterB->first)
      ++iterA;
    // iterB is strictly before iterA => increment iterB
    else if (iterA->first > iterB->second)
      ++iterB;
    // iterB is at the start of iterA, but iterA has some values that go
    // beyond those of iterB. We have to set the lower bound of iterA to the
    // upper bound of iterB + 1 and increment iterB.
    // A(3, 100) - B(3, 5) => A(6,100)
    else if (iterA->first == iterB->first && iterA->first <= iterB->second &&
             iterA->second > iterB->second) {
      iterA->first = iterB->second + 1;
      ++iterB;
    }
    // iterB is at the end of iterA, but iterA has some values that come
    // before iterB. We have to set the upper bound of iterA to the lower
    // bound of iterB - 1 and increment both iterators.
    // A(4, 50) - B(40, 50) => A(4, 39)
    else if (iterA->second >= iterB->first && iterA->second == iterB->second &&
             iterA->first < iterB->first) {
      iterA->second = iterB->first - 1;
      ++iterA;
      ++iterB;
    }
    // iterB is in the middle of iterA. We have to split iterA and increment
    // iterB.
    // A(2, 10) B(5, 7) => (2, 4), (8, 10)
    else if (iterA->first < iterB->first && iterA->second > iterB->second) {
      iterA->first = iterB->second + 1;
      iterA = a.insert(iterA, UseInterval(iterA->first, iterB->first - 1)) + 1;
      ++iterB;
    }
    // Both intervals are equal. We have to erase the whole interval.
    // A(5, 5) B(5, 5) => {}
    else {
      iterA = a.erase(iterA);
      ++iterB;
    }
  }
}

/// This performs an interval union of two sorted interval vectors.
/// Return false if there is an interval interference.
std::pair<UserangeAnalysis::IntervalVector, bool>
UserangeAnalysis::intervalUnion(IntervalVector intervalA,
                                IntervalVector intervalB) const {
  IntervalVector intervalUnion;
  auto iterA = intervalA.begin();
  auto iterB = intervalB.begin();
  // Iterate over both interval vectors simultaneously.
  for (; iterA != intervalA.end() && iterB != intervalB.end();) {
    // iterA comes before iterB.
    if (iterA->first < iterB->first && iterA->second < iterB->first)
      intervalUnion.push_back(*iterA++);
    // iterB comes before iterA.
    else if (iterB->first < iterA->first && iterB->second < iterA->first)
      intervalUnion.push_back(*iterB++);
    // There is an interval interference. We thus have to return false.
    else
      return std::pair<IntervalVector, bool>(intervalUnion, false);
  }
  // Push the remaining intervals.
  for (; iterA != intervalA.end(); ++iterA)
    intervalUnion.push_back(*iterA);
  for (; iterB != intervalB.end(); ++iterB)
    intervalUnion.push_back(*iterB);

  // Merge consecutive intervals that have no gap between each other.
  for (auto it = intervalUnion.begin(); it != intervalUnion.end() - 1;) {
    if (it->second == (it + 1)->first - 1) {
      (it + 1)->first = it->first;
      it = intervalUnion.erase(it);
    } else
      ++it;
  }
  return std::pair<IntervalVector, bool>(intervalUnion, true);
}

void UserangeAnalysis::print(raw_ostream &os) {
  os << "// ---- UserangeAnalysis -----\n";
  std::vector<Value> values;
  for (auto const &item : useIntervalMap) {
    values.push_back(item.first);
  }
  std::sort(values.begin(), values.end(), [&](Value left, Value right) {
    if (left.getDefiningOp()) {
      if (right.getDefiningOp())
        return operationIds[left.getDefiningOp()] <
               operationIds[right.getDefiningOp()];
      else
        return true;
    }
    if (right.getDefiningOp())
      return false;
    return operationIds[&left.getParentBlock()->front()] <
           operationIds[&right.getParentBlock()->front()];
  });
  for (auto value : values) {
    os << "Value: " << value << (value.getDefiningOp() ? "\n" : "");
    auto rangeIt = useIntervalMap[value].begin();
    os << "Userange: {(" << rangeIt->first << ", " << rangeIt->second << ")";
    for (rangeIt++; rangeIt != useIntervalMap[value].end(); ++rangeIt) {
      os << ", (" << rangeIt->first << ", " << rangeIt->second << ")";
    }
    os << "}\n";
  }
  os << "// ---------------------------\n";
}