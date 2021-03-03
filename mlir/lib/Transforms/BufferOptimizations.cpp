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
#include "mlir/Analysis/UserangeAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"

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
        postDominators(op), userange(op, allocs, aliases) {}

  /// Reuses already allocated buffers to save allocation operations.
  void reuse(Operation *operation) {
    // Create a list of values that can potentially be replaced for each value
    // in the useRangeMap. The potentialReuseMap maps each value to the
    // respective list.
    llvm::MapVector<Value, SmallVector<Value, 4>> potentialReuseMap;
    for (BufferPlacementAllocs::AllocEntry entry : allocs) {
      Value itemA = std::get<0>(entry);
      SmallVector<Value, 4> potReuseVector;
      for (BufferPlacementAllocs::AllocEntry entry : allocs) {
        Value itemB = std::get<0>(entry);
        // Do not compare an item to itself and make sure that the value of item
        // B is not a BlockArgument. BlockArguments cannot be reused. Also
        // perform a type check.
        if (itemA == itemB || !checkTypeCompatibility(itemA, itemB))
          continue;

        // Check if itemA can replace itemB.
        if (!userange.rangesInterfere(itemA, itemB))
          continue;

        // Get the defining operation of itemA.
        Block *defOpBlock = itemA.isa<BlockArgument>()
                                ? itemA.getParentBlock()
                                : itemA.getDefiningOp()->getBlock();

        // The defining block of itemA has to dominate all uses of itemB.
        if (!dominatesAllUses(defOpBlock, itemB))
          continue;

        // Insert itemB into the right place of the potReuseVector. The order of
        // the vector is defined via the program order of the first use of each
        // item.
        auto it = potReuseVector.begin();
        while (it != potReuseVector.end()) {
          if (userange.getFirstUseIndex(itemB) <
              userange.getFirstUseIndex(*it)) {
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

    /// Maps a value to the set of values that it replaces.
    llvm::MapVector<Value, DenseSet<Value>> actualReuseMap;

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

          // Check if the replaced value already replaces other values and also
          // add them to the reused set.
          if (actualReuseMap.count(v)) {
            actualReuseMap[item].insert(actualReuseMap[v].begin(),
                                        actualReuseMap[v].end());
            actualReuseMap.erase(v);
          }

          // Merge the userange of v into the userange of item.
          userange.unionRanges(item, v);

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

          if (replacedSet.contains(*potReuseValue) ||
              transitiveInterference(*potReuseValue, potReuses,
                                     actualReuseMap) ||
              !userange.rangesInterfere(item, *potReuseValue))
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
  /// Check if all uses of item are dominated by the given block.
  bool dominatesAllUses(Block *block, Value item) {
    for (OpOperand &operand : item.getUses()) {
      if (!dominators.dominates(block, operand.getOwner()->getBlock()))
        return false;
    }
    return true;
  }

  /// Checks if there is a transitive interference between potReuseValue and the
  /// value that may replace it, we call this value V. potReuses is the vector
  /// of all values that can potentially be replaced by V. If potReuseValue
  /// already replaces any other value that is not part of the potReuses vector
  /// it cannot be replaced by V anymore.
  bool transitiveInterference(
      Value potReuseValue, SmallVector<Value, 4> *potReuses,
      llvm::MapVector<Value, DenseSet<Value>> &actualReuseMap) {
    return actualReuseMap.count(potReuseValue) &&
           llvm::any_of(actualReuseMap[potReuseValue], [&](Value vReuse) {
             return !std::count(potReuses->begin(), potReuses->end(), vReuse);
           });
  }

  /// Checks if the types of the given values are compatible for a
  /// replacement.
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

    // We need the actual alloc operation of both types. For aliases we need
    // to check for the defining OP of the alias' origin.
    Operation *defOpA = a.getDefiningOp();
    Operation *defOpB = b.getDefiningOp();

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

  /// The current dominance info.
  DominanceInfo dominators;

  /// The current postdominance info.
  PostDominanceInfo postDominators;

  /// The current userange info.
  UserangeAnalysis userange;

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
    Operation *funcOp = getFunction();
    BufferReuse optimizer(funcOp);
    optimizer.reuse(funcOp);
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
