// RUN: mlir-opt -copy-removal -split-input-file %s
//| FileCheck %s

// All linalg copies except the linalg.copy(%1, %9) must be removed since the
// defining operation of %1 and its DeallocOp have been defined in another block.

// CHECK-LABEL: func @nested_region_control_flow_div_nested
func @nested_region_control_flow_div_nested(%arg0: index, %arg1: index) -> memref<?x?xf32> {
  %0 = cmpi eq, %arg0, %arg1 : index
  %1 = alloc(%arg0, %arg0) : memref<?x?xf32>
  // CHECK: %{{.*}} = scf.if
  %2 = scf.if %0 -> (memref<?x?xf32>) {
    // CHECK: %[[PERCENT3:.*]] = scf.if
    %3 = scf.if %0 -> (memref<?x?xf32>) {
      %c0_0 = constant 0 : index
      %7 = dim %1, %c0_0 : memref<?x?xf32>
      %c1_1 = constant 1 : index
      %8 = dim %1, %c1_1 : memref<?x?xf32>
      %9 = alloc(%7, %8) : memref<?x?xf32>
      // CHECK: linalg.copy({{.*}}, %[[PERCENT9:.*]])
      linalg.copy(%1, %9) : memref<?x?xf32>, memref<?x?xf32>
      // CHECK: scf.yield %[[PERCENT9]]
      scf.yield %9 : memref<?x?xf32>
    } else {
      // CHECK: %[[PERCENT7:.*]] = alloc
      %7 = alloc(%arg0, %arg1) : memref<?x?xf32>
      %c0_0 = constant 0 : index
      %8 = dim %7, %c0_0 : memref<?x?xf32>
      %c1_1 = constant 1 : index
      %9 = dim %7, %c1_1 : memref<?x?xf32>
      // CHECK-NOT: %{{.*}} = alloc
      // CHECK-NOT: linalg.copy(%[[PERCENT7]], %{{.*}})
      // CHECK-NOT: dealloc %[[PERCENT7]]
      %10 = alloc(%8, %9) : memref<?x?xf32>
      linalg.copy(%7, %10) : memref<?x?xf32>, memref<?x?xf32>
      dealloc %7 : memref<?x?xf32>
      // CHECK: scf.yield %[[PERCENT7]]
      scf.yield %10 : memref<?x?xf32>
    }
    %c0 = constant 0 : index
    %4 = dim %3, %c0 : memref<?x?xf32>
    %c1 = constant 1 : index
    %5 = dim %3, %c1 : memref<?x?xf32>
    // CHECK-NOT: %{{.*}} = alloc
    // CHECK-NOT: linalg.copy(%[[PERCENT3]], %{{.*}})
    // CHECK-NOT: dealloc %[[PERCENT3]]
    %6 = alloc(%4, %5) : memref<?x?xf32>
    linalg.copy(%3, %6) : memref<?x?xf32>, memref<?x?xf32>
    dealloc %3 : memref<?x?xf32>
    // CHECK: scf.yield %[[PERCENT3]]
    scf.yield %6 : memref<?x?xf32>
  } else {
    // CHECK: %[[PERCENT3:.*]] = alloc
    %3 = alloc(%arg1, %arg1) : memref<?x?xf32>
    %c0 = constant 0 : index
    %4 = dim %3, %c0 : memref<?x?xf32>
    %c1 = constant 1 : index
    %5 = dim %3, %c1 : memref<?x?xf32>
    // CHECK-NOT: %{{.*}} = alloc
    // CHECK-NOT: linalg.copy(%[[PERCENT3]], %{{.*}})
    // CHECK-NOT: dealloc %[[PERCENT3]]
    %6 = alloc(%4, %5) : memref<?x?xf32>
    linalg.copy(%3, %6) : memref<?x?xf32>, memref<?x?xf32>
    dealloc %3 : memref<?x?xf32>
    // CHECK: scf.yield %[[PERCENT3]]
    scf.yield %6 : memref<?x?xf32>
  }
  dealloc %1 : memref<?x?xf32>
  return %2 : memref<?x?xf32>
}

// -----

// CHECK-LABEL: func @simple_test
func @simple_test() -> memref<5xf32> {
  %temp = alloc() : memref<5xf32>
  %ret = alloc() : memref<5xf32>
  linalg.copy(%ret, %temp) : memref<5xf32>, memref<5xf32>
  dealloc %ret : memref<5xf32>
  return %temp : memref<5xf32>
}
// CHECK-SAME: () -> memref<5xf32>
// CHECK-NEXT: %[[ret:.*]] = alloc()
// CHECK-NOT: linalg.copy(%[[ret]], %{{.*}})
// CHECK-NOT: dealloc %[[ret]]
// CHECK: return %[[ret]]

// -----

// It is legal to remove the copy operation that %ret has a usage before the copy
// operation. The allocation of %temp and the deallocation of %ret should be also
// removed.

// CHECK-LABEL: func @test_with_ret_usage_before_copy
func @test_with_ret_usage_before_copy() -> memref<5xf32> {
  %ret = alloc() : memref<5xf32>
  %temp = alloc() : memref<5xf32>
  %c0 = constant 0 : index
  %dimension = dim %ret, %c0 : memref<5xf32>
  linalg.copy(%ret, %temp) : memref<5xf32>, memref<5xf32>
  dealloc %ret : memref<5xf32>
  return %temp : memref<5xf32>
}
// CHECK-NEXT: %[[ret:.*]] = alloc()
// CHECK-NOT: %{{.*}} = alloc
// CHECK-NEXT: %{{.*}} = constant
// CHECK-NEXT: %[[DIM:.*]] = dim %[[ret]]
// CHECK-NOT: linalg.copy(%[[ret]], %{{.*}})
// CHECK-NOT: dealloc %[[ret]]
// CHECK: return %[[ret]]

// -----

// It is illegal to remove a copy operation that %ret has a usage after copy
// operation.

// CHECK-LABEL: func @test_with_ret_usage_after_copy
func @test_with_ret_usage_after_copy() -> memref<5xf32> {
  %ret = alloc() : memref<5xf32>
  %temp = alloc() : memref<5xf32>
  // CHECK: linalg.copy
  linalg.copy(%ret, %temp) : memref<5xf32>, memref<5xf32>
  %c0 = constant 0 : index
  %dimension = dim %ret, %c0 : memref<5xf32>
  dealloc %ret : memref<5xf32>
  return %temp : memref<5xf32>
}

// -----

// It is illegal to remove a copy operation that %temp has a usage before copy
// operation.

// CHECK-LABEL: func @test_with_temp_usage_before_copy
func @test_with_temp_usage_before_copy() -> memref<5xf32> {
  %ret = alloc() : memref<5xf32>
  %temp = alloc() : memref<5xf32>
  %c0 = constant 0 : index
  %dimension = dim %temp, %c0 : memref<5xf32>
  // CHECK: linalg.copy
  linalg.copy(%ret, %temp) : memref<5xf32>, memref<5xf32>
  dealloc %ret : memref<5xf32>
  return %temp : memref<5xf32>
}

// -----

// It is legal to remove the copy operation that %temp has a usage after the copy
// operation. The allocation of %temp and the deallocation of %ret could be also
// removed.

// However the following pattern is not handled by copy removal.
//   %from = alloc()
//   %to = alloc()
//   copy(%from, %to)
//   read_from(%from) + write_to(%something_else)
//   dealloc(%from)
//   return %to
// In particular, linalg.generic is a memoryEffectOp between copy and dealloc.
// Since no alias analysis is performed and no distinction is made between reads
// and writes, the linalg.generic with effects blocks copy removal.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @test_with_temp_usage_after_copy
func @test_with_temp_usage_after_copy() -> memref<5xf32> {
  %ret = alloc() : memref<5xf32>
  %res = alloc() : memref<5xf32>
  %temp = alloc() : memref<5xf32>
  linalg.copy(%ret, %temp) : memref<5xf32>, memref<5xf32>
  linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]}
    ins(%temp : memref<5xf32>)
   outs(%res : memref<5xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }
  dealloc %ret : memref<5xf32>
  return %temp : memref<5xf32>
}
// CHECK-NEXT: %[[ret:.*]] = alloc()
// CHECK-NEXT: %[[res:.*]] = alloc()
// CHECK-NEXT: %[[temp:.*]] = alloc()
// CHECK-NEXT: linalg.copy(%[[ret]], %[[temp]])
// CHECK-NEXT: linalg.generic
//      CHECK: dealloc %[[ret]]
//      CHECK: return %[[temp]]

// -----

// CHECK-LABEL: func @make_allocation
func @make_allocation() -> memref<5xf32> {
  %mem = alloc() : memref<5xf32>
  return %mem : memref<5xf32>
}

// CHECK-LABEL: func @test_with_function_call
func @test_with_function_call() -> memref<5xf32> {
  // CHECK-NEXT: %[[ret:.*]] = call @make_allocation() : () -> memref<5xf32>
  %ret = call @make_allocation() : () -> (memref<5xf32>)
  // CHECK-NOT: %{{.*}} = alloc
  // CHECK-NOT: linalg.copy(%[[ret]], %{{.*}})
  // CHECK-NOT: dealloc %[[ret]]
  %temp = alloc() : memref<5xf32>
  linalg.copy(%ret, %temp) : memref<5xf32>, memref<5xf32>
  dealloc %ret : memref<5xf32>
  // CHECK: return %[[ret]]
  return %temp : memref<5xf32>
}

// -----

// CHECK-LABEL: func @multiple_deallocs_in_different_blocks
func @multiple_deallocs_in_different_blocks(%cond : i1) -> memref<5xf32> {
  // CHECK-NEXT: %[[PERCENT0:.*]] = alloc()
  %0 = alloc() : memref<5xf32>
  cond_br %cond, ^bb1, ^bb2
^bb1:
  dealloc %0 : memref<5xf32>
  // CHECK: br ^[[BB3:.*]](%[[PERCENT0]]
  br ^bb3(%0 : memref<5xf32>)
^bb2:
  // CHECK-NOT: %{{.*}} = alloc
  // CHECK-NOT: linalg.copy(%[[PERCENT0]], %{{.*}})
  // CHECK-NOT: dealloc %[[PERCENT0]]
  %temp = alloc() : memref<5xf32>
  linalg.copy(%0, %temp) : memref<5xf32>, memref<5xf32>
  dealloc %0 : memref<5xf32>
  // CHECK: br ^[[BB3]](%[[PERCENT0]]
  br ^bb3(%temp : memref<5xf32>)
^bb3(%res : memref<5xf32>):
  return %res : memref<5xf32>
}

// -----

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @test_ReuseCopyTargetAsSource
func @test_ReuseCopyTargetAsSource(%arg0: memref<2xf32>, %result: memref<2xf32>){
  // CHECK-SAME: (%[[ARG0:.*]]: memref<2xf32>, %[[RES:.*]]: memref<2xf32>)
  // CHECK-NOT: %{{.*}} = alloc
  %temp = alloc() : memref<2xf32>
  // CHECK-NEXT: linalg.generic
  // CHECK-SAME: ins(%[[ARG0]]{{.*}}outs(%[[RES]]
  // CHECK-NOT: linalg.copy(%{{.*}}, %[[RES]])
  // CHECK-NOT: dealloc %{{.*}}
  linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]}
    ins(%arg0 : memref<2xf32>)
   outs(%temp : memref<2xf32>) {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }
  "linalg.copy"(%temp, %result) : (memref<2xf32>, memref<2xf32>) -> ()
  dealloc %temp : memref<2xf32>
  // CHECK: return
  return
}

// -----

// Copy operation must not be removed since an operation writes to %to value
// before copy.

#map0 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @test_ReuseCopyTargetAsSource
func @test_ReuseCopyTargetAsSource(%arg0: memref<2xf32>){
  %to = alloc() : memref<2xf32>
  %temp = alloc() : memref<2xf32>
  linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]}
    ins(%arg0 : memref<2xf32>)
   outs(%temp : memref<2xf32>) {
  ^bb0(%gen1_arg0: f32, %gen1_arg1: f32):
    %tmp1 = exp %gen1_arg0 : f32
    linalg.yield %tmp1 : f32
  }
  linalg.generic {
    indexing_maps = [#map0, #map0],
    iterator_types = ["parallel"]}
    ins(%arg0 : memref<2xf32>)
   outs(%to : memref<2xf32>) {
  ^bb0(%gen2_arg0: f32, %gen2_arg1: f32):
    %tmp2 = exp %gen2_arg0 : f32
    linalg.yield %tmp2 : f32
  }
  // CHECK: linalg.copy
  "linalg.copy"(%temp, %to) : (memref<2xf32>, memref<2xf32>) -> ()
  dealloc %temp : memref<2xf32>
  return
}

// -----

// The only redundant copy is linalg.copy(%4, %5)

// CHECK-LABEL: func @loop_alloc
func @loop_alloc(%arg0: index, %arg1: index, %arg2: index, %arg3: memref<2xf32>, %arg4: memref<2xf32>) {
  // CHECK: %{{.*}} = alloc()
  %0 = alloc() : memref<2xf32>
  dealloc %0 : memref<2xf32>
  // CHECK: %{{.*}} = alloc()
  %1 = alloc() : memref<2xf32>
  // CHECK: linalg.copy
  linalg.copy(%arg3, %1) : memref<2xf32>, memref<2xf32>
  %2 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %1) -> (memref<2xf32>) {
    %3 = cmpi eq, %arg5, %arg1 : index
    // CHECK: dealloc
    dealloc %arg6 : memref<2xf32>
    // CHECK: %[[PERCENT4:.*]] = alloc()
    %4 = alloc() : memref<2xf32>
    // CHECK-NOT: alloc
    // CHECK-NOT: linalg.copy
    // CHECK-NOT: dealloc
    %5 = alloc() : memref<2xf32>
    linalg.copy(%4, %5) : memref<2xf32>, memref<2xf32>
    dealloc %4 : memref<2xf32>
    // CHECK: %[[PERCENT6:.*]] = alloc()
    %6 = alloc() : memref<2xf32>
    // CHECK: linalg.copy(%[[PERCENT4]], %[[PERCENT6]])
    linalg.copy(%5, %6) : memref<2xf32>, memref<2xf32>
    scf.yield %6 : memref<2xf32>
  }
  // CHECK: linalg.copy
  linalg.copy(%2, %arg4) : memref<2xf32>, memref<2xf32>
  dealloc %2 : memref<2xf32>
  return
}

// -----

// The linalg.copy operation can be removed in addition to alloc and dealloc
// operations. All uses of %0 is then replaced with %arg2.

// CHECK-LABEL: func @check_with_affine_dialect
func @check_with_affine_dialect(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xf32>) {
  // CHECK-SAME: (%[[ARG0:.*]]: memref<4xf32>, %[[ARG1:.*]]: memref<4xf32>, %[[RES:.*]]: memref<4xf32>)
  // CHECK-NOT: alloc
  %0 = alloc() : memref<4xf32>
  affine.for %arg3 = 0 to 4 {
    %5 = affine.load %arg0[%arg3] : memref<4xf32>
    %6 = affine.load %arg1[%arg3] : memref<4xf32>
    %7 = cmpf ogt, %5, %6 : f32
    // CHECK: %[[SELECT_RES:.*]] = select
    %8 = select %7, %5, %6 : f32
    // CHECK-NEXT: affine.store %[[SELECT_RES]], %[[RES]]
    affine.store %8, %0[%arg3] : memref<4xf32>
  }
  // CHECK-NOT: linalg.copy
  // CHECK-NOT: dealloc
  "linalg.copy"(%0, %arg2) : (memref<4xf32>, memref<4xf32>) -> ()
  dealloc %0 : memref<4xf32>
  //CHECK: return
  return
}
