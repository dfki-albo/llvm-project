// RUN: mlir-opt %s -test-print-userange -split-input-file 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : func_empty
func @func_empty() {
  return
}
//      CHECK:  ---- UserangeAnalysis -----
// CHECK-NEXT:  ---------------------------

// -----

// CHECK-LABEL: Testing : useRangeGap
func @useRangeGap(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
  test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
  br ^bb3
^bb2:
  test.buffer_based in(%arg2: memref<2xf32>) out(%0: memref<2xf32>)
  test.buffer_based in(%arg2: memref<2xf32>) out(%1: memref<2xf32>)
  br ^bb3
^bb3:
  return
}
//      CHECK:  Value: %0 {{ *}}
// CHECK-NEXT:  Userange: {(3, 3), (6, 6)}
//      CHECK:  Value: %1 {{ *}}
// CHECK-NEXT:  Userange: {(4, 4), (7, 7)}

// -----

// CHECK-LABEL: Testing : loopWithNestedRegion
func @loopWithNestedRegion(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = alloc() : memref<2xf32>
  %1 = alloc() : memref<2xf32>
  %2 = alloc() : memref<2xf32>
  %3 = alloc() : memref<2xf32>
  br ^bb1
^bb1:
  %4 = scf.if %arg0 -> (memref<2xf32>) {
    test.buffer_based in(%arg1: memref<2xf32>) out(%0: memref<2xf32>)
    scf.yield %2 : memref<2xf32>
  } else {
    test.buffer_based in(%arg1: memref<2xf32>) out(%1: memref<2xf32>)
    scf.yield %2 : memref<2xf32>
  }
  br ^bb2
^bb2:
  cond_br %arg0, ^bb1, ^bb3
^bb3:
  test.buffer_based in(%arg1: memref<2xf32>) out(%2: memref<2xf32>)
  test.buffer_based in(%arg1: memref<2xf32>) out(%3: memref<2xf32>)
  return
}
//      CHECK:  Value: %0 {{ *}}
// CHECK-NEXT:  Userange: {(5, 11)}
//      CHECK:  Value: %1 {{ *}}
// CHECK-NEXT:  Userange: {(5, 11)}
//      CHECK:  Value: %2 {{ *}}
// CHECK-NEXT:  Userange: {(5, 12)}
//      CHECK:  Value: %3 {{ *}}
// CHECK-NEXT:  Userange: {(13, 13)}
//      CHECK:  Value: %4 {{ *}}
//      CHECK:  Userange: {(9, 9)}

// -----

// CHECK-LABEL: Testing : condBranchWithAlias
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
  br ^bb4(%0 : memref<2xf32>)
^bb4(%5 : memref<2xf32>):
  test.copy(%5, %arg2) : (memref<2xf32>, memref<2xf32>)
  return
}
//      CHECK:  Value: %0 {{ *}}
// CHECK-NEXT:  Userange: {(2, 3), (7, 13)}
//      CHECK:  Value: %1 {{ *}}
// CHECK-NEXT:  Userange: {(5, 8)}
//      CHECK:  Value: %3 {{ *}}
// CHECK-NEXT:  Userange: {(9, 9)}
//      CHECK:  Value: %4 {{ *}}
// CHECK-NEXT:  Userange: {(11, 11)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-NEXT:  Userange: {(7, 8)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-NEXT:  Userange: {(13, 13)}