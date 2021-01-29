//===- BufferOptimizations.cpp - pre-pass optimizations for bufferization -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for three optimization passes. The first two
// passes try to move alloc nodes out of blocks to reduce the number of
// allocations and copies during buffer deallocation. The third pass tries to
// convert heap-based allocations to stack-based allocations, if possible.

#include "PassDetail.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;

/// Returns true if the given operation implements a known high-level region-
/// based control-flow interface.
static bool isKnownControlFlowInterface(Operation *op) {
  return isa<LoopLikeOpInterface, RegionBranchOpInterface>(op);
}

/// Check if the size of the allocation is less than the given size. The
/// transformation is only applied to small buffers since large buffers could
/// exceed the stack space.
static bool isSmallAlloc(Value alloc, unsigned maximumSizeInBytes,
                         unsigned bitwidthOfIndexType,
                         unsigned maxRankOfAllocatedMemRef) {
  auto type = alloc.getType().dyn_cast<ShapedType>();
  if (!type || !alloc.getDefiningOp<AllocOp>())
    return false;
  if (!type.hasStaticShape()) {
    // Check if the dynamic shape dimension of the alloc is produced by RankOp.
    // If this is the case, it is likely to be small. Furthermore, the dimension
    // is limited to the maximum rank of the allocated memref to avoid large
    // values by multiplying several small values.
    if (type.getRank() <= maxRankOfAllocatedMemRef) {
      return llvm::all_of(
          alloc.getDefiningOp()->getOperands(),
          [&](Value operand) { return operand.getDefiningOp<RankOp>(); });
    }
    return false;
  }
  // For index types, use the provided size, as the type does not know.
  unsigned int bitwidth = type.getElementType().isIndex()
                              ? bitwidthOfIndexType
                              : type.getElementTypeBitWidth();
  return type.getNumElements() * bitwidth <= maximumSizeInBytes * 8;
}

/// Checks whether the given aliases leave the allocation scope.
static bool
leavesAllocationScope(Region *parentRegion,
                      const BufferAliasAnalysis::ValueSetT &aliases) {
  for (Value alias : aliases) {
    for (auto *use : alias.getUsers()) {
      // If there is at least one alias that leaves the parent region, we know
      // that this alias escapes the whole region and hence the associated
      // allocation leaves allocation scope.
      if (use->hasTrait<OpTrait::ReturnLike>() &&
          use->getParentRegion() == parentRegion)
        return true;
    }
  }
  return false;
}

/// Checks, if an automated allocation scope for a given alloc value exists.
static bool hasAllocationScope(Value alloc,
                               const BufferAliasAnalysis &aliasAnalysis) {
  Region *region = alloc.getParentRegion();
  do {
    if (Operation *parentOp = region->getParentOp()) {
      // Check if the operation is an automatic allocation scope and whether an
      // alias leaves the scope. This means, an allocation yields out of
      // this scope and can not be transformed in a stack-based allocation.
      if (parentOp->hasTrait<OpTrait::AutomaticAllocationScope>() &&
          !leavesAllocationScope(region, aliasAnalysis.resolve(alloc)))
        return true;
      // Check if the operation is a known control flow interface and break the
      // loop to avoid transformation in loops. Furthermore skip transformation
      // if the operation does not implement a RegionBeanchOpInterface.
      if (BufferPlacementTransformationBase::isLoop(parentOp) ||
          !isKnownControlFlowInterface(parentOp))
        break;
    }
  } while ((region = region->getParentRegion()));
  return false;
}

/// Checks if a given operation uses a value.
static bool isRealUse(Value value, Operation *op) {
  return llvm::any_of(op->getOperands(),
                      [&](Value operand) { return operand == value; });
}

namespace {

//===----------------------------------------------------------------------===//
// BufferAllocationHoisting
//===----------------------------------------------------------------------===//

/// A base implementation compatible with the `BufferAllocationHoisting` class.
struct BufferAllocationHoistingStateBase {
  /// A pointer to the current dominance info.
  DominanceInfo *dominators;

  /// The current allocation value.
  Value allocValue;

  /// The current placement block (if any).
  Block *placementBlock;

  /// Initializes the state base.
  BufferAllocationHoistingStateBase(DominanceInfo *dominators, Value allocValue,
                                    Block *placementBlock)
      : dominators(dominators), allocValue(allocValue),
        placementBlock(placementBlock) {}
};

/// Implements the actual hoisting logic for allocation nodes.
template <typename StateT>
class BufferAllocationHoisting : public BufferPlacementTransformationBase {
public:
  BufferAllocationHoisting(Operation *op)
      : BufferPlacementTransformationBase(op), dominators(op),
        postDominators(op) {}

  /// Moves allocations upwards.
  void hoist() {
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);
      Operation *definingOp = allocValue.getDefiningOp();
      assert(definingOp && "No defining op");
      auto operands = definingOp->getOperands();
      auto resultAliases = aliases.resolve(allocValue);
      // Determine the common dominator block of all aliases.
      Block *dominatorBlock =
          findCommonDominator(allocValue, resultAliases, dominators);
      // Init the initial hoisting state.
      StateT state(&dominators, allocValue, allocValue.getParentBlock());
      // Check for additional allocation dependencies to compute an upper bound
      // for hoisting.
      Block *dependencyBlock = nullptr;
      if (!operands.empty()) {
        // If this node has dependencies, check all dependent nodes with respect
        // to a common post dominator. This ensures that all dependency values
        // have been computed before allocating the buffer.
        ValueSetT dependencies(std::next(operands.begin()), operands.end());
        dependencyBlock = findCommonDominator(*operands.begin(), dependencies,
                                              postDominators);
      }

      // Find the actual placement block and determine the start operation using
      // an upper placement-block boundary. The idea is that placement block
      // cannot be moved any further upwards than the given upper bound.
      Block *placementBlock = findPlacementBlock(
          state, state.computeUpperBound(dominatorBlock, dependencyBlock));
      Operation *startOperation = BufferPlacementAllocs::getStartOperation(
          allocValue, placementBlock, liveness);

      // Move the alloc in front of the start operation.
      Operation *allocOperation = allocValue.getDefiningOp();
      allocOperation->moveBefore(startOperation);
    }
  }

private:
  /// Finds a valid placement block by walking upwards in the CFG until we
  /// either cannot continue our walk due to constraints (given by the StateT
  /// implementation) or we have reached the upper-most dominator block.
  Block *findPlacementBlock(StateT &state, Block *upperBound) {
    Block *currentBlock = state.placementBlock;
    // Walk from the innermost regions/loops to the outermost regions/loops and
    // find an appropriate placement block that satisfies the constraint of the
    // current StateT implementation. Walk until we reach the upperBound block
    // (if any).

    // If we are not able to find a valid parent operation or an associated
    // parent block, break the walk loop.
    Operation *parentOp;
    Block *parentBlock;
    while ((parentOp = currentBlock->getParentOp()) &&
           (parentBlock = parentOp->getBlock()) &&
           (!upperBound ||
            dominators.properlyDominates(upperBound, currentBlock))) {
      // Try to find an immediate dominator and check whether the parent block
      // is above the immediate dominator (if any).
      DominanceInfoNode *idom = dominators.getNode(currentBlock)->getIDom();
      if (idom && dominators.properlyDominates(parentBlock, idom->getBlock())) {
        // If the current immediate dominator is below the placement block, move
        // to the immediate dominator block.
        currentBlock = idom->getBlock();
        state.recordMoveToDominator(currentBlock);
      } else {
        // We have to move to our parent block since an immediate dominator does
        // either not exist or is above our parent block. If we cannot move to
        // our parent operation due to constraints given by the StateT
        // implementation, break the walk loop. Furthermore, we should not move
        // allocations out of unknown region-based control-flow operations.
        if (!isKnownControlFlowInterface(parentOp) ||
            !state.isLegalPlacement(parentOp))
          break;
        // Move to our parent block by notifying the current StateT
        // implementation.
        currentBlock = parentBlock;
        state.recordMoveToParent(currentBlock);
      }
    }
    // Return the finally determined placement block.
    return state.placementBlock;
  }

  /// The dominator info to find the appropriate start operation to move the
  /// allocs.
  DominanceInfo dominators;

  /// The post dominator info to move the dependent allocs in the right
  /// position.
  PostDominanceInfo postDominators;

  /// The map storing the final placement blocks of a given alloc value.
  llvm::DenseMap<Value, Block *> placementBlocks;
};

/// A state implementation compatible with the `BufferAllocationHoisting` class
/// that hoists allocations into dominator blocks while keeping them inside of
/// loops.
struct BufferAllocationHoistingState : BufferAllocationHoistingStateBase {
  using BufferAllocationHoistingStateBase::BufferAllocationHoistingStateBase;

  /// Computes the upper bound for the placement block search.
  Block *computeUpperBound(Block *dominatorBlock, Block *dependencyBlock) {
    // If we do not have a dependency block, the upper bound is given by the
    // dominator block.
    if (!dependencyBlock)
      return dominatorBlock;

    // Find the "lower" block of the dominator and the dependency block to
    // ensure that we do not move allocations above this block.
    return dominators->properlyDominates(dominatorBlock, dependencyBlock)
               ? dependencyBlock
               : dominatorBlock;
  }

  /// Returns true if the given operation does not represent a loop.
  bool isLegalPlacement(Operation *op) {
    return !BufferPlacementTransformationBase::isLoop(op);
  }

  /// Sets the current placement block to the given block.
  void recordMoveToDominator(Block *block) { placementBlock = block; }

  /// Sets the current placement block to the given block.
  void recordMoveToParent(Block *block) { recordMoveToDominator(block); }
};

/// A state implementation compatible with the `BufferAllocationHoisting` class
/// that hoists allocations out of loops.
struct BufferAllocationLoopHoistingState : BufferAllocationHoistingStateBase {
  using BufferAllocationHoistingStateBase::BufferAllocationHoistingStateBase;

  /// Remembers the dominator block of all aliases.
  Block *aliasDominatorBlock;

  /// Computes the upper bound for the placement block search.
  Block *computeUpperBound(Block *dominatorBlock, Block *dependencyBlock) {
    aliasDominatorBlock = dominatorBlock;
    // If there is a dependency block, we have to use this block as an upper
    // bound to satisfy all allocation value dependencies.
    return dependencyBlock ? dependencyBlock : nullptr;
  }

  /// Returns true if the given operation represents a loop and one of the
  /// aliases caused the `aliasDominatorBlock` to be "above" the block of the
  /// given loop operation. If this is the case, it indicates that the
  /// allocation is passed via a back edge.
  bool isLegalPlacement(Operation *op) {
    return BufferPlacementTransformationBase::isLoop(op) &&
           !dominators->dominates(aliasDominatorBlock, op->getBlock());
  }

  /// Does not change the internal placement block, as we want to move
  /// operations out of loops only.
  void recordMoveToDominator(Block *block) {}

  /// Sets the current placement block to the given block.
  void recordMoveToParent(Block *block) { placementBlock = block; }
};

//===----------------------------------------------------------------------===//
// BufferPlacementPromotion
//===----------------------------------------------------------------------===//

/// Promotes heap-based allocations to stack-based allocations (if possible).
class BufferPlacementPromotion : BufferPlacementTransformationBase {
public:
  BufferPlacementPromotion(Operation *op)
      : BufferPlacementTransformationBase(op) {}

  /// Promote buffers to stack-based allocations.
  void promote(unsigned maximumSize, unsigned bitwidthOfIndexType,
               unsigned maxRankOfAllocatedMemRef) {
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value alloc = std::get<0>(entry);
      Operation *dealloc = std::get<1>(entry);
      // Checking several requirements to transform an AllocOp into an AllocaOp.
      // The transformation is done if the allocation is limited to a given
      // size. Furthermore, a deallocation must not be defined for this
      // allocation entry and a parent allocation scope must exist.
      if (!isSmallAlloc(alloc, maximumSize, bitwidthOfIndexType,
                        maxRankOfAllocatedMemRef) ||
          dealloc || !hasAllocationScope(alloc, aliases))
        continue;

      Operation *startOperation = BufferPlacementAllocs::getStartOperation(
          alloc, alloc.getParentBlock(), liveness);
      // Build a new alloca that is associated with its parent
      // `AutomaticAllocationScope` determined during the initialization phase.
      OpBuilder builder(startOperation);
      Operation *allocOp = alloc.getDefiningOp();
      Operation *alloca = builder.create<AllocaOp>(
          alloc.getLoc(), alloc.getType().cast<MemRefType>(),
          allocOp->getOperands());

      // Replace the original alloc by a newly created alloca.
      allocOp->replaceAllUsesWith(alloca);
      allocOp->erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// BufferReuse
//===----------------------------------------------------------------------===//

/// Reuses already allocated buffer to save allocation operations.
class BufferReuse : BufferPlacementTransformationBase {
public:
  BufferReuse(Operation *op)
      : BufferPlacementTransformationBase(op), dominators(op),
        postDominators(op) {}

  /// An implementation for the first and last use of a value.
  struct FirstAndLastUse {
    Operation *firstUse;
    Operation *lastUse;

    bool operator==(const FirstAndLastUse &other) const {
      return firstUse == other.firstUse && lastUse == other.lastUse;
    }

    bool operator!=(const FirstAndLastUse &other) const {
      return firstUse != other.firstUse || lastUse != other.lastUse;
    }
  };

  /// Reuses already allocated buffers to save allocation operations.
  void reuse() {
    // Find all first and last uses for all allocated values and their aliases
    // and save them in the useRangeMap.
    llvm::MapVector<Value, FirstAndLastUse> useRangeMap;
    for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
      Value allocValue = std::get<0>(entry);

      // Resolve all aliases for the allocValue to later save them in a cache.
      ValueSetT aliasSet = aliases.resolve(allocValue);

      // Iterate over the aliasSet and compute the use range.
      for (Value aliasValue : aliasSet) {
        if (aliasValue != allocValue && !aliasOfMap.count(aliasValue))
          aliasOfMap.insert(std::pair<Value, Value>(aliasValue, allocValue));

        // Check if the allocValue/alias is already processed or has no users.
        if (useRangeMap.count(aliasValue) || aliasValue.use_empty())
          continue;

        FirstAndLastUse firstAndLastUse{};
        // Iterate over all uses of the allocValue/alias and find their first
        // and last use.
        for (Operation *user : aliasValue.getUsers()) {
          // No update is needed if the operation has already been considered.
          if (firstAndLastUse.firstUse == user ||
              firstAndLastUse.lastUse == user)
            continue;

          updateFirstOp(aliasValue, firstAndLastUse.firstUse, user);

          updateLastOp(aliasValue, firstAndLastUse.lastUse, user);
        }
        useRangeMap.insert(
            std::pair<Value, FirstAndLastUse>(aliasValue, firstAndLastUse));
      }

      // Remove the allocValue from its own aliasList to prevent reflexive
      // checks and ensure correct behavior after we insert the aliases of the
      // reused buffer.
      aliasSet.erase(allocValue);
      aliasCache.insert(std::pair<Value, ValueSetT>(allocValue, aliasSet));
    }

    // Create a list of values that can potentially be replaced for each value
    // in the useRangeMap. The potentialReuseMap maps each value to the
    // respective list.
    llvm::MapVector<Value, SmallVector<Value, 4>> potentialReuseMap;

    for (auto const &useRangeItemA : useRangeMap) {
      SmallVector<Value, 4> potReuseVector;
      Value itemA = useRangeItemA.first;
      for (auto const &useRangeItemB : useRangeMap) {
        Value itemB = useRangeItemB.first;
        // Do not compare an item to itself and make sure that the value of item
        // B is not a BlockArgument. BlockArguments cannot be reused. Also
        // perform a type check.
        if (useRangeItemA == useRangeItemB || aliasOfMap.count(itemB) ||
            !checkTypeCompatibility(itemA, itemB))
          continue;

        FirstAndLastUse usesA = useRangeItemA.second;
        FirstAndLastUse usesB = useRangeItemB.second;

        // Check if itemA can replace itemB.
        if (!isReusePossible(itemA, itemB, usesA, usesB))
          continue;

        // Get the defining operation of itemA.
        Block *defOpBlock = itemA.isa<BlockArgument>()
                                ? itemA.getParentBlock()
                                : itemA.getDefiningOp()->getBlock();

        // The defining OP of itemA has to dominate the first use of itemB.
        if (!dominators.dominates(defOpBlock, usesB.firstUse->getBlock()))
          continue;

        // Insert itemB into the right place of the potReuseVector. The order of
        // the vector is defined via the program order of the first use of each
        // item.
        auto it = potReuseVector.begin();
        while (it != potReuseVector.end()) {
          if (isUsedBefore(useRangeMap[itemB].firstUse,
                           useRangeMap[*it].firstUse)) {
            potReuseVector.insert(it, itemB);
            break;
          }
          ++it;
        }
        if (it == potReuseVector.end())
          potReuseVector.push_back(itemB);
      }

      potentialReuseMap.insert(
          std::pair<Value, SmallVector<Value, 4>>(itemA, potReuseVector));
    }

    // The replacedSet contains all values that are going to be replaced.
    DenseSet<Value> replacedSet;
    // The currentReuserSet contains all values that are replacing another
    // value in the current iteration. Note: This is necessary because the
    // replacing property is not transitive.
    DenseSet<Value> currentReuserSet;
    // Fixpoint iteration over the potential reuses.
    for (;;) {
      // Clear the currentReuserSet for this iteration.
      currentReuserSet.clear();
      // Step 1 of the fixpoint iteration: Choose a value to be replaced for
      // each value in the potentialReuseMap.
      for (auto &potReuser : potentialReuseMap) {
        Value item = potReuser.first;
        SmallVector<Value, 4> potReuses = potReuser.second;

        // If the current value is replaced already we have to skip it.
        if (replacedSet.contains(item))
          continue;

        // Find a value that can be reused. If the value is already in the
        // currentReuserSet then we have to break. Due to the order of the
        // values we must not skip it, because it can potentially be replaced in
        // the next iteration. However, we may skip the value if it is replaced
        // by another value.
        for (Value v : potReuses) {
          if (currentReuserSet.contains(v))
            break;
          if (replacedSet.contains(v))
            continue;

          // Update the actualReuseMap.
          actualReuseMap[item].insert(v);

          // Join the aliases of the reusee and reuser.
          llvm::set_union(aliasCache[item], aliasCache[v]);

          // Check if the replaced value already replaces other values and also
          // add them to the reused set.
          if (actualReuseMap.count(v)) {
            actualReuseMap[item].insert(actualReuseMap[v].begin(),
                                        actualReuseMap[v].end());
            actualReuseMap.erase(v);
          }

          Operation *itemLastUse = useRangeMap[item].lastUse;
          Operation *vLastUse = useRangeMap[v].lastUse;
          // If itemLastUse and vLast are in different branches set the last use
          // to the next Postdominator. If they are in neighboring nested
          // regions set the last use to the common parent op. Otherwise, update
          // the last use to the last used operation of the replaced value.
          if (!isUsedBefore(itemLastUse, vLastUse) &&
              !isUsedBefore(vLastUse, itemLastUse)) {
            // As we are updating the last OP here we have to erase the value
            // from this set as the new lastOP might not be inside a nested
            // region anymore.
            lastUseInsideNestedRegion.erase(item);
            useRangeMap[item].lastUse =
                findOpInDominator(postDominators, item, itemLastUse, vLastUse);
          } else
            useRangeMap[item].lastUse = vLastUse;

          currentReuserSet.insert(item);
          replacedSet.insert(v);
          break;
        }
      }

      // If the currentReuseSet is empty we can terminate the fixpoint
      // iteration.
      if (currentReuserSet.empty())
        break;

      // Step 2 of the fixpoint iteration: Update the potentialReuseVectors for
      // each value in the potentialReuseMap. Due to the chosen replacements in
      // step 1 some values might not be replaceable anymore. Also remove all
      // replaced values from the potentialReuseMap.
      for (auto itReuseMap = potentialReuseMap.begin();
           itReuseMap != potentialReuseMap.end();) {
        Value item = itReuseMap->first;
        FirstAndLastUse uses = useRangeMap[item];
        SmallVector<Value, 4> *potReuses = &itReuseMap->second;

        // If the item is already reused, we can remove it from the
        // potentialReuseMap.
        if (replacedSet.contains(item)) {
          potentialReuseMap.erase(itReuseMap);
          continue;
        }

        // Iterate over the potential reuses and check if they can still be
        // reused.
        for (Value *potReuseValue = potReuses->begin();
             potReuseValue != potReuses->end();) {
          FirstAndLastUse potReuseUses = useRangeMap[*potReuseValue];

          if (replacedSet.contains(*potReuseValue) ||
              transitiveInterference(*potReuseValue, potReuses) ||
              !isReusePossible(item, *potReuseValue, uses, potReuseUses))
            potReuses->erase(potReuseValue);
          else
            ++potReuseValue;
        }
        ++itReuseMap;
      }
    }

    // Delete the alloc of the value that is replaced and replace all uses of
    // that value.
    for (auto &reuse : actualReuseMap) {
      for (Value reuseValue : reuse.second) {
        reuseValue.replaceAllUsesWith(reuse.first);
        reuseValue.getDefiningOp()->erase();
      }
    }
  }

private:
  /// Checks if there is a transitive interference between potReuseValue and the
  /// value that may replace it, we call this value V. potReuses is the vector
  /// of all values that can potentially be replaced by V. If potReuseValue
  /// already replaces any other value that is not part of the potReuses vector
  /// it cannot be replaced by V anymore.
  bool transitiveInterference(Value potReuseValue,
                              SmallVector<Value, 4> *potReuses) {
    return actualReuseMap.count(potReuseValue) &&
           llvm::any_of(actualReuseMap[potReuseValue], [&](Value vReuse) {
             return !std::count(potReuses->begin(), potReuses->end(), vReuse);
           });
  }

  /// Check if a reuse of two values and their first and last uses is possible.
  /// It depends on userange interferences, alias interference and real uses.
  /// Returns true if a reuse is possible.
  bool isReusePossible(Value itemA, Value itemB, FirstAndLastUse usesA,
                       FirstAndLastUse usesB) {
    // Check if the last use of itemA is before the first use of itemB.
    if (isUsedBefore(usesA.lastUse, usesB.firstUse)) {
      // If this is the case we need to check if itemB is used after the
      // introduction of an alias of itemA. Should this be the case we must not
      // replace it with itemA as there might be an alias interference. The
      // actual check happens when the respective alias is itemA.
      if (isUsedAfterAliasIntroduction(itemA, usesB.firstUse))
        return false;
    }
    // Check if the first use of itemB is not before the last use of itemA.
    // This is the case if the two uses are either in neighbor branches or if
    // they are the same OP.
    else if (!isUsedBefore(usesB.firstUse, usesA.lastUse)) {
      // If they are the same OP we can still replace itemB with itemA if the OP
      // is not a ``real use''. This is the case if we had to choose a postDom
      // OP as the last use for itemA.
      if (usesA.lastUse == usesB.firstUse) {
        if (lastUseInsideNestedRegion.contains(itemA) ||
            isRealUse(itemA, usesA.lastUse))
          return false;
        return true;
      }
      // If one of the two uses spawns a nested region and other item is inside
      // of it we must not replace itemB with itemA. If one of the two uses
      // equals the ancestor OP of the respective other use we know that this is
      // the case.
      Operation *ancOpB =
          usesA.lastUse->getBlock()->findAncestorOpInBlock(*usesB.firstUse);
      Operation *ancOpA =
          usesB.firstUse->getBlock()->findAncestorOpInBlock(*usesA.lastUse);
      if (ancOpB == usesA.lastUse || ancOpA == usesB.firstUse)
        return false;
    } else
      return false;
    return true;
  }

  /// Check if the given operation is used after the introduction of one of the
  /// aliases of the given value.
  bool isUsedAfterAliasIntroduction(Value v, Operation *otherFirstUse) {
    return llvm::any_of(aliasCache[v], [&](Value alias) {
      Operation *firstOpInBlock = &alias.getParentBlock()->front();
      return isUsedBefore(firstOpInBlock, otherFirstUse) ||
             alias.getParentBlock() == otherFirstUse->getBlock();
    });
  }

  /// Updates the first Operation from the two given ones.
  void updateFirstOp(Value value, Operation *&op, Operation *user) {
    if (!op || isUsedBefore(user, op))
      op = user;
    else if (!isUsedBefore(op, user) && !isUsedBefore(user, op))
      op = findOpInDominator(dominators, value, op, user);
  }

  /// Updates the last Operation from the two given ones.
  void updateLastOp(Value value, Operation *&op, Operation *user) {
    // As we are updating the last OP here we have to erase value from this set
    // as the new lastOP might not be inside a nested region anymore.
    lastUseInsideNestedRegion.erase(value);
    if (!op || isUsedBefore(op, user))
      op = user;
    else if (!isUsedBefore(op, user) && !isUsedBefore(user, op))
      op = findOpInDominator(postDominators, value, op, user);
  }

  /// Given two OPs, this finds the correct OP to use as the next first or last
  /// use (depending on postDom) of the value belonging to op.
  template <template <bool> class DominanceInfoT, bool IsPostDom>
  Operation *findOpInDominator(const DominanceInfoT<IsPostDom> &dominance,
                               Value value, Operation *op, Operation *user) {
    Block *dominator =
        dominance.findNearestCommonDominator(op->getBlock(), user->getBlock());
    // If no ancestorOp exists in the dominator we know that the two OPs are in
    // neighboring branches and we thus return the respective dominator
    // operation. Otherwise we know that the two OPs are in neighboring regions
    // and thus need to  return the ancestorOp.
    Operation *ancestorOp = dominator->findAncestorOpInBlock(*op);
    if (!ancestorOp)
      return IsPostDom ? &dominator->front() : &dominator->back();
    // If we are searching for a last op we have to remember that the actual
    // last use of value was inside a nested region and we chose its ancestor op
    // as the new last use.
    if (IsPostDom)
      lastUseInsideNestedRegion.insert(value);
    return ancestorOp;
  }

  /// Returns true if op is used before other.
  bool isUsedBefore(Operation *op, Operation *other) {
    Block *opBlock = op->getBlock();
    Block *otherBlock = other->getBlock();

    // Both Operations are in the same block.
    if (opBlock == otherBlock)
      return op->isBeforeInBlock(other);

    // Check if op is used in a dominator of other. If other is inside a nested
    // region we need to find its ancestor Op and check if it is after op.
    if (dominators.dominates(opBlock, otherBlock)) {
      Operation *ancestor = opBlock->findAncestorOpInBlock(*other);
      return !ancestor || op->isBeforeInBlock(ancestor);
    }

    // Should one of the OPs (or both) be inside a nested region we need to
    // bring the respective blocks to the top level. If we pass by the common
    // dominator in both cases we can use the two ancestor OPs in this dominator
    // to determine which OP comes first.
    Block *commonDom =
        dominators.findNearestCommonDominator(opBlock, otherBlock);
    if (findBlocksOnCommonLevel(op, other, opBlock, otherBlock, commonDom)) {
      Operation *opAncestor = commonDom->findAncestorOpInBlock(*op);
      Operation *otherAncestor = commonDom->findAncestorOpInBlock(*other);
      return opAncestor->isBeforeInBlock(otherAncestor);
    }

    // Recursive call to find if the otherBlock is a successor of opBlock. The
    // common postdominator is used as a termination condition.
    Block *postDom =
        postDominators.findNearestCommonDominator(opBlock, otherBlock);
    SmallPtrSet<Block *, 6> visited;
    return isSuccessor(opBlock, otherBlock, postDom, visited);
  }

  /// Set block to the block of the parent op one level. Returns true if the
  /// dominator block was passed by.
  bool findBlocksOnCommonLevel(Operation *op, Operation *otherOp, Block *&block,
                               Block *&otherBlock, Block *commonDom) {

    DenseMap<Operation *, Block *> visitedOps;
    bool domFound = false;
    // Climb the parentOps until no parentOp exists and update block to the
    // block below the parentOp of op. Insert each pair of visted parentOp and
    // block in the map.
    while (op) {
      block = op->getBlock();
      op = op->getParentOp();
      domFound |= block == commonDom;
      visitedOps.insert(std::pair<Operation *, Block *>(op, block));
    }
    // Climb the parentOps until no parentOp exists and update otherBlock to the
    // block below the parentOp of otherOp. If the parentOp was already visted,
    // set block to the corresponding block of that op.
    while (otherOp) {
      otherBlock = otherOp->getBlock();
      otherOp = otherOp->getParentOp();
      domFound |= block == commonDom;
      if (visitedOps.count(otherOp)) {
        block = visitedOps[otherOp];
        return domFound;
      }
    }
    return domFound;
  }

  /// Recursive function that returns true if the target Block is reachable from
  /// the currentBlock.
  bool isSuccessor(Block *currentBlock, Block *target, Block *postDom,
                   SmallPtrSet<Block *, 6> &visited) {
    if (currentBlock == target)
      return true;
    if (currentBlock == postDom)
      return false;
    for (Block *succ : currentBlock->getSuccessors()) {
      if (visited.insert(succ).second &&
          isSuccessor(succ, target, postDom, visited))
        return true;
    }
    return false;
  }

  /// Checks if the types of the given values are compatible for a replacement.
  bool checkTypeCompatibility(Value a, Value b) {
    auto shapedA = a.getType().cast<ShapedType>();
    auto shapedB = b.getType().cast<ShapedType>();

    // If both types are shaped we can check for equality.
    if (shapedA.hasStaticShape() && shapedB.hasStaticShape())
      return a.getType() == b.getType();
    // If only one of the types is shaped we cannot detect compatibility since
    // we do not know how the allocation operation behaves on its operands.
    if (shapedA.hasStaticShape() != shapedB.hasStaticShape())
      return false;

    // We need the actual alloc operation of both types. For aliases we need to
    // check for the defining OP of the alias' origin.
    Operation *defOpA = getAliasDefiningOp(a);
    Operation *defOpB = getAliasDefiningOp(b);

    assert(defOpA && "Defining OP must not be null.");
    assert(defOpB && "Defining OP must not be null.");

    // If the alloc method or the number of operands is not the same the types
    // cannot be compatible.
    if (defOpA->getName() != defOpB->getName() ||
        defOpA->getNumOperands() != defOpB->getNumOperands())
      return false;

    // If all operands are equal the types are compatible.
    for (auto const &pair :
         llvm::zip(defOpA->getOperands(), defOpB->getOperands())) {
      if (std::get<0>(pair) != std::get<1>(pair))
        return false;
    }
    return true;
  }

  // Returns the defining operation of the value. If the value is an alias then
  // return the defining operation of its origin.
  Operation *getAliasDefiningOp(Value value) {
    auto iter = aliasOfMap.find(value);
    return iter != aliasOfMap.end() ? iter->second.getDefiningOp()
                                    : value.getDefiningOp();
  }

  /// Cache the alias lists for all values to avoid the recomputation.
  BufferAliasAnalysis::ValueMapT aliasCache;

  /// Maps an alias to its origin. This is used to compute the defining OP of an
  /// alias.
  llvm::DenseMap<Value, Value> aliasOfMap;

  /// The current dominance info.
  DominanceInfo dominators;

  /// The current postdominance info.
  PostDominanceInfo postDominators;

  /// Maps a value to the set of values that it replaces.
  llvm::MapVector<Value, DenseSet<Value>> actualReuseMap;

  /// A set of all values which lastUse is an unreal use spawning a nested
  /// region.
  llvm::DenseSet<Value> lastUseInsideNestedRegion;
};

//===----------------------------------------------------------------------===//
// BufferOptimizationPasses
//===----------------------------------------------------------------------===//

/// The buffer hoisting pass that hoists allocation nodes into dominating
/// blocks.
struct BufferHoistingPass : BufferHoistingBase<BufferHoistingPass> {

  void runOnFunction() override {
    // Hoist all allocations into dominator blocks.
    BufferAllocationHoisting<BufferAllocationHoistingState> optimizer(
        getFunction());
    optimizer.hoist();
  }
};

/// The buffer loop hoisting pass that hoists allocation nodes out of loops.
struct BufferLoopHoistingPass : BufferLoopHoistingBase<BufferLoopHoistingPass> {

  void runOnFunction() override {
    // Hoist all allocations out of loops.
    BufferAllocationHoisting<BufferAllocationLoopHoistingState> optimizer(
        getFunction());
    optimizer.hoist();
  }
};

/// The promote buffer to stack pass that tries to convert alloc nodes into
/// alloca nodes.
struct PromoteBuffersToStackPass
    : PromoteBuffersToStackBase<PromoteBuffersToStackPass> {

  PromoteBuffersToStackPass(unsigned maxAllocSizeInBytes,
                            unsigned bitwidthOfIndexType,
                            unsigned maxRankOfAllocatedMemRef) {
    this->maxAllocSizeInBytes = maxAllocSizeInBytes;
    this->bitwidthOfIndexType = bitwidthOfIndexType;
    this->maxRankOfAllocatedMemRef = maxRankOfAllocatedMemRef;
  }

  void runOnFunction() override {
    // Move all allocation nodes and convert candidates into allocas.
    BufferPlacementPromotion optimizer(getFunction());
    optimizer.promote(this->maxAllocSizeInBytes, this->bitwidthOfIndexType,
                      this->maxRankOfAllocatedMemRef);
  }
};

/// The buffer reuse pass that uses already allocated buffers if all critera
/// are met.
struct BufferReusePass : BufferReuseBase<BufferReusePass> {

  void runOnFunction() override {
    // Reuse allocated buffer instead of new allocation.
    BufferReuse optimizer(getFunction());
    optimizer.reuse();
  }
};

} // end anonymous namespace

std::unique_ptr<Pass> mlir::createBufferHoistingPass() {
  return std::make_unique<BufferHoistingPass>();
}

std::unique_ptr<Pass> mlir::createBufferLoopHoistingPass() {
  return std::make_unique<BufferLoopHoistingPass>();
}

std::unique_ptr<Pass>
mlir::createPromoteBuffersToStackPass(unsigned maxAllocSizeInBytes,
                                      unsigned bitwidthOfIndexType,
                                      unsigned maxRankOfAllocatedMemRef) {
  return std::make_unique<PromoteBuffersToStackPass>(
      maxAllocSizeInBytes, bitwidthOfIndexType, maxRankOfAllocatedMemRef);
}

std::unique_ptr<Pass> mlir::createBufferReusePass() {
  return std::make_unique<BufferReusePass>();
}
