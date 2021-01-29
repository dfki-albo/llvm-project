// RUN: mlir-opt -buffer-reuse -split-input-file %s | FileCheck %s

// Expected behavior: %0 replaces %1 and %2 replaces %3 and %4. %0 and %1 do not
// interfere with each other despite their shared alias. %0 could in theory
// safely replace %3 and %4. However, its alias %2 is already present in the
// respective block and should thus be chosen as the replacement.
// CHECK-LABEL: func @condBranchWithAlias
func @condBranchWithAlias(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb3(%0 : memref<2xf32>)
^bb2:
  %1 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
  br ^bb3(%1 : memref<2xf32>)
^bb3(%2 : memref<2xf32>):
  %3 = alloc() : memref<2xf32>
  test.copy(%2, %arg2) : (memref<2xf32>, memref<2xf32>)
  test.copy(%3, %arg2) : (memref<2xf32>, memref<2xf32>)
  %4 = alloc() : memref<2xf32>
  test.copy(%4, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB3:.*]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB3]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB3]](%[[BLOCKARG0:.*]]: {{.*}}):
// CHECK-NEXT: test.copy(%[[BLOCKARG0]], %[[ARG2]])
// CHECK-NEXT: test.copy(%[[BLOCKARG0]], %[[ARG2]])
// CHECK-NEXT: test.copy(%[[BLOCKARG0]], %[[ARG2]])
// CHECK-NEXT: return

// -----

// Expected behavior: This code only needs a single alloc.
// CHECK-LABEL: func @allReuseSimple
func @allReuseSimple(%arg0: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  %2 = alloc() : memref<2xf32>
  %3 = alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  test.buffer_based in(%arg0: memref<2xf32>) out(%1: memref<2xf32>)
  test.buffer_based in(%arg0: memref<2xf32>) out(%2: memref<2xf32>)
  test.buffer_based in(%arg0: memref<2xf32>) out(%3: memref<2xf32>)
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: return

// -----

// Expected behavior: %0 can't replace %1 as its alloc OP does not dominate the
// first use of %1.
// CHECK-LABEL: func @allocDominance
func @allocDominance(%arg0: i1, %arg1: memref<2xf32>) {
  cond_br %arg0, ^bb1, ^bb2
 ^bb1:
  %0 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb2
 ^bb2:
  %1 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB2]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: return

// -----

// Expected behavior: Nothing can be replaced as there is an alias interference.
// CHECK-LABEL: func @aliasInterference
func @aliasInterference(%arg0: i1, %arg1: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb1(%0 : memref<2xf32>)
^bb1(%2 : memref<2xf32>):
  test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
  test.buffer_based in(%arg1: memref<2xf32>) out(%2: memref<2xf32>)
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB1:.*]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB1]](%[[BLOCKARG0:.*]]: {{.*}}):
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[BLOCKARG0]]
// CHECK-NEXT: return

// -----

// Expected behavior: %1 should be replaced by %2 as there is no interference.
// %2 should be the replacer here as all uses of %1 come after its introduction
// CHECK-LABEL: func @aliasReuse
func @aliasReuse(%arg0: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb1(%0 : memref<2xf32>)
^bb1(%2 : memref<2xf32>):
  test.buffer_based in(%arg0: memref<2xf32>) out(%2: memref<2xf32>)
  test.buffer_based in(%arg0: memref<2xf32>) out(%1: memref<2xf32>)
  return
}
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB1:.*]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB1]](%[[BLOCKARG0:.*]]: {{.*}}):
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[BLOCKARG0]]
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[BLOCKARG0]]
// CHECK-NEXT: return

// -----

// Expected behavior: %0 should replace %1. We don't have a clear last use of %0
// here so we use the first OP of the common post dominator as its ``last use''.
// However, this should not prevent the replacement of any buffers after said
// post dominator block.
// CHECK-LABEL: func @unrealUseInPostDom
func @unrealUseInPostDom(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  test.buffer_based in(%0: memref<2xf32>) out(%arg1: memref<2xf32>)
  br ^bb3
^bb2:
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb3
^bb3:
  cond_br %arg0, ^bb4, ^bb5
^bb4:
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  br ^bb6
^bb5:
  test.copy(%1, %arg2) : (memref<2xf32>, memref<2xf32>)
  br ^bb6
^bb6:
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC0]]{{.*}}out(%[[ARG1]]
// CHECK-NEXT: br ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB3]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB4:.*]], ^[[BB5:.*]]
//      CHECK: ^[[BB4]]:
// CHECK-NEXT: test.copy(%[[ALLOC0]], %[[ARG2]])
// CHECK-NEXT: br ^[[BB6:.*]]
//      CHECK: ^[[BB5]]:
// CHECK-NEXT: test.copy(%[[ALLOC0]], %[[ARG2]])
// CHECK-NEXT: br ^[[BB6]]
// CHECK-NEXT: ^[[BB6]]:
// CHECK-NEXT: return

// -----

// Expected behavior: Nothing should be replaced as both buffers interfere
// within a single OP.
// CHECK-LABEL: func @sameOperation
func @sameOperation() {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  test.copy(%1, %0) : (memref<2xf32>, memref<2xf32>)
  return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: test.copy(%[[ALLOC1]], %[[ALLOC0]])
// CHECK-NEXT: return

// -----

// Expected behavior: %0 replaces %1. Due to the order of the alloc operations
// and the first uses %1 will be the first buffer in the potential replacement
// list of %0. After it is replaced the last use of %0 will be first OP of bb3
// (the post dominator of %0 and %1). %2 thus can't be replaced anymore.
// CHECK-LABEL: func @branchReuse
func @branchReuse(%arg0: i1, %arg1: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  %2 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  test.buffer_based in(%0: memref<2xf32>) out(%arg1: memref<2xf32>)
  test.buffer_based in(%arg1: memref<2xf32>) out(%2: memref<2xf32>)
  br ^bb3
^bb2:
  test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
  br ^bb3
^bb3:
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC0]]{{.*}}out(%[[ARG1]]
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: br ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB3]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: return

// -----

// Expected behavior: No replacement due to the type mismatch.
// CHECK-LABEL: func @typeMismatch
func @typeMismatch(%arg0: memref<2xf32>, %arg1: memref<4xf16>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<4xf16>
  test.buffer_based in(%arg0: memref<2xf32>) out(%0: memref<2xf32>)
  test.buffer_based in(%arg1: memref<4xf16>) out(%1: memref<4xf16>)
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: test.buffer_based in(%[[ARG0]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: return

// -----

// Expected behavior: %5 replaces %2 due to the same allocation with the same
// operands, otherwise no replacement due to the type mismatch.
// CHECK-LABEL: func @complexTypeMatching
func @complexTypeMatching(%arg0: i1, %arg1: index, %arg2: index, %arg3 : index) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc(%arg1) : memref<?xf32>
  %2 = alloc(%arg2) : memref<?xf32>
  %3 = alloc(%arg1, %arg2) : memref<?x?xf32>
  %4 = alloc(%arg1, %arg3) : memref<?x?xf32>
  %5 = alloc(%arg2) : memref<?xf32>
  test.buffer_based in(%0: memref<2xf32>) out(%0: memref<2xf32>)
  cond_br %arg0, ^bb1, ^bb4
^bb1:
  test.buffer_based in(%1: memref<?xf32>) out(%1: memref<?xf32>)
  cond_br %arg0, ^bb2, ^bb3
^bb2:
  test.buffer_based in(%3: memref<?x?xf32>) out(%3: memref<?x?xf32>)
  test.buffer_based in(%4: memref<?x?xf32>) out(%4: memref<?x?xf32>)
  br ^bb4
^bb3:
  test.buffer_based in(%5: memref<?xf32>) out(%5: memref<?xf32>)
  br ^bb2
^bb4:
  test.buffer_based in(%2: memref<?xf32>) out(%2: memref<?xf32>)
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}},
// CHECK-SAME: %[[ARG3:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc(%[[ARG1]])
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc(%[[ARG1]], %[[ARG2]])
// CHECK-NEXT: %[[ALLOC3:.*]] = alloc(%[[ARG1]], %[[ARG3]])
// CHECK-NEXT: %[[ALLOC4:.*]] = alloc(%[[ARG2]])
// CHECK-NEXT: test.buffer_based in(%[[ALLOC0]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB4:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB2:.*]], ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC2]]{{.*}}out(%[[ALLOC2]]
// CHECK-NEXT: test.buffer_based in(%[[ALLOC3]]{{.*}}out(%[[ALLOC3]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC4]]{{.*}}out(%[[ALLOC4]]
//      CHECK: ^[[BB4]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC4]]{{.*}}out(%[[ALLOC4]]
// CHECK-NEXT: return

// -----

// Expected behavior: In this case %2 can replace %0 and %0 can replace %1.
// However, %2 can't replace %1. Due to the ordering the only valid replacement
// is %0 replaces %2.
// CHECK-LABEL: func @nonTransitive
func @nonTransitive(%arg0: i1, %arg1: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  %2 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
 ^bb1:
  test.buffer_based in(%arg1: memref<2xf32>) out(%2: memref<2xf32>)
  test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
  test.buffer_based in(%arg1: memref<2xf32>) out(%2: memref<2xf32>)
  br ^bb3
 ^bb2:
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  br ^bb3
 ^bb3:
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
// CHECK-NEXT: br ^[[BB3]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: return

// -----

// Expected behavior: %0 replaces %1 even though %0 is inside a nesteg region.
// CHECK-LABEL: func @condBranchNestedRegion
func @condBranchNestedRegion(%arg0: i1, %arg1: memref<2xf32>,
                              %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%arg2: memref<2xf32>) {
    ^nb1:
      test.buffer_based in(%arg2: memref<2xf32>) out(%0: memref<2xf32>)
      test.region_yield %arg1: memref<2xf32>
  }
  br ^bb3
^bb2:
  test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
  br ^bb3
^bb3:
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.region_buffer_based in(%[[ARG1]]{{.*}}out(%[[ARG2]]
//      CHECK: test.buffer_based in(%[[ARG2]]{{.*}}out(%[[ALLOC0]]
//      CHECK: br ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
//      CHECK: br ^[[BB3:.*]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: return

// -----

// Expected behavior: %0 replaces %1. Nested regions should not prevent
// replacement.
// CHECK-LABEL: func @condBranchNestedRegions
func @condBranchNestedRegions(%arg0: i1, %arg1: memref<2xf32>,
                              %arg2: memref<2xf32>) {
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%arg2: memref<2xf32>) {
    ^nb1:
      test.buffer_based in(%arg2: memref<2xf32>) out(%0: memref<2xf32>)
      test.region_yield %arg1: memref<2xf32>
  }
  br ^bb3
^bb2:
  test.region_buffer_based in(%arg1: memref<2xf32>) out(%arg2: memref<2xf32>) {
    ^nb2:
      test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
      test.region_yield %arg1: memref<2xf32>
  }
  br ^bb3
^bb3:
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.region_buffer_based in(%[[ARG1]]{{.*}}out(%[[ARG2]]
//      CHECK: test.buffer_based in(%[[ARG2]]{{.*}}out(%[[ALLOC0]]
//      CHECK: br ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: test.region_buffer_based in(%[[ARG1]]{{.*}}out(%[[ARG2]]
//      CHECK: test.buffer_based in(%[[ARG1]]{{.*}}out(%[[ALLOC0]]
//      CHECK: br ^[[BB3:.*]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: return

// -----

// Expected behavior: %2 replaces %5. %1 cannot replace %2 or %5 due to
// the uses in both neighboring nested regions.
// CHECK-LABEL: func @nestedRegionControlFlow
func @nestedRegionControlFlow(%arg0 : index, %arg1 : index) -> memref<2xf32> {
  %0 = cmpi "eq", %arg0, %arg1 : index
  %1 = alloc() : memref<2xf32>
  %2 = alloc() : memref<2xf32>
  %3 = scf.if %0 -> (memref<2xf32>) {
    %4 = scf.if %0 -> (memref<2xf32>) {
      scf.yield %1 : memref<2xf32>
    } else {
      %5 = alloc() : memref<2xf32>
      scf.yield %5 : memref<2xf32>
    }
    scf.yield %4 : memref<2xf32>
  } else {
    test.buffer_based in(%1: memref<2xf32>) out(%1: memref<2xf32>)
    scf.yield %2 : memref<2xf32>
  }
  return %3 : memref<2xf32>
}

//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC3:.*]] = scf.if
// CHECK-NEXT: %[[ALLOC4:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC1]]
//      CHECK: scf.yield %[[ALLOC2]]
//      CHECK: scf.yield %[[ALLOC4]]
//      CHECK: test.buffer_based in(%[[ALLOC1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: scf.yield %[[ALLOC2]]
//      CHECK: return %[[ALLOC3]]

// -----

// Expected behavior: Nothing should be reused here.
// CHECK-LABEL: func @nestedRegionControlFlowNoReuse
func @nestedRegionControlFlowNoReuse(%arg0 : index,
                                     %arg1 : index) -> memref<2xf32> {
  %0 = cmpi "eq", %arg0, %arg1 : index
  %1 = alloc() : memref<2xf32>
  %2 = alloc() : memref<2xf32>
  %3 = scf.if %0 -> (memref<2xf32>) {
    %4 = scf.if %0 -> (memref<2xf32>) {
      scf.yield %1 : memref<2xf32>
    } else {
      scf.yield %2 : memref<2xf32>
    }
    scf.yield %4 : memref<2xf32>
  } else {
    test.buffer_based in(%1: memref<2xf32>) out(%2: memref<2xf32>)
    scf.yield %2 : memref<2xf32>
  }
  return %3 : memref<2xf32>
}

//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: %[[ALLOC3:.*]] = scf.if
// CHECK-NEXT: %[[ALLOC4:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC1]]
//      CHECK: scf.yield %[[ALLOC2]]
//      CHECK: scf.yield %[[ALLOC4]]
//      CHECK: test.buffer_based in(%[[ALLOC1]]{{.*}}out(%[[ALLOC2]]
// CHECK-NEXT: scf.yield %[[ALLOC2]]
//      CHECK: return %[[ALLOC3]]

// -----

// Expected behavior: A control flow inside a nested region. %1 should replace
// %2.
// CHECK-LABEL: func @ControlFlowInsideNestedRegion
func @ControlFlowInsideNestedRegion(%arg0 : i1) {
  %0 = alloc() : memref<2xf32>
  test.region_buffer_based in(%0: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = alloc() : memref<2xf32>
    %2 = alloc() : memref<2xf32>
    cond_br %arg0, ^bb1, ^bb2
    ^bb1:
      test.buffer_based in(%2: memref<2xf32>) out(%2: memref<2xf32>)
      br ^bb3
    ^bb2:
      br ^bb3
    ^bb3:
      test.buffer_based in(%1: memref<2xf32>) out(%1: memref<2xf32>)
    %tmp1 = exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: test.region_buffer_based in(%[[ALLOC0]]{{.*}}out(%[[ALLOC0]]
//      CHECK: %[[ALLOC1:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: br ^[[BB3:.*]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC1]]{{.*}}out(%[[ALLOC1]]
//      CHECK: return

// -----

// Expected behavior: A control flow inside a nested region with no possilbe
// reuse.
// CHECK-LABEL: func @ControlFlowInsideNestedRegion
func @ControlFlowInsideNestedRegionNoReuse(%arg0 : i1) {
  %0 = alloc() : memref<2xf32>
  test.region_buffer_based in(%0: memref<2xf32>) out(%0: memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %1 = alloc() : memref<2xf32>
    %2 = alloc() : memref<2xf32>
    cond_br %arg0, ^bb1, ^bb2
    ^bb1:
      test.buffer_based in(%2: memref<2xf32>) out(%2: memref<2xf32>)
      br ^bb3
    ^bb2:
      br ^bb3
    ^bb3:
      test.buffer_based in(%1: memref<2xf32>) out(%1: memref<2xf32>)
      test.buffer_based in(%2: memref<2xf32>) out(%0: memref<2xf32>)
    %tmp1 = exp %gen1_arg0 : f32
    test.region_yield %tmp1 : f32
  }
  return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
//      CHECK: %[[ALLOC0:.*]] = alloc()
// CHECK-NEXT: test.region_buffer_based in(%[[ALLOC0]]{{.*}}out(%[[ALLOC0]]
//      CHECK: %[[ALLOC1:.*]] = alloc()
//      CHECK: %[[ALLOC2:.*]] = alloc()
// CHECK-NEXT: cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC2]]{{.*}}out(%[[ALLOC2]]
// CHECK-NEXT: br ^[[BB3:.*]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: test.buffer_based in(%[[ALLOC1]]{{.*}}out(%[[ALLOC1]]
// CHECK-NEXT: test.buffer_based in(%[[ALLOC2]]{{.*}}out(%[[ALLOC0]]
//      CHECK: return