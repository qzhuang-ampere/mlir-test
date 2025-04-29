// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize -cse | FileCheck %s

// This is a simple tile-and-fuse example with a single fusion group.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

module {
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<1x4096xf32>, %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2 = linalg.matmul ins(%arg2, %arg0 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%1 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg1 : tensor<4096x4096xf32>, tensor<1x4096xf32>) outs(%0 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<4096x4096xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4096x4096xf32>) outs(%0 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.cmpf ugt, %in, %cst : f32
      %6 = arith.select %5, %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<4096x4096xf32>
    return %4 : tensor<4096x4096xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %csimd_width = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_row = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_col = transform.param.constant 2048 : i64 -> !transform.param<i64>

      %mat = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      //transform.print %mat : !transform.any_op

      %tiled_op_mat, %forall_op_mat:2 = transform.structured.tile_using_for %mat tile_sizes [%ct_row, %ct_col]
           : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      %gens = transform.structured.match ops{["linalg.generic", "linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      //transform.print %gens : !transform.any_op

      transform.foreach  %gens : !transform.any_op {
      ^bb1(%gen : !transform.any_op):
        %tiled_op, %forloop_op:2 = transform.structured.tile_using_for %gen tile_sizes [%ct_row, %ct_col]
            : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        //transform.print %forloop_op#0 : !transform.any_op
        //transform.print %tiled_op : !transform.any_op

        %tiled_op1, %forall_op = transform.structured.tile_using_for %tiled_op tile_sizes [0, %csimd_width]
            : (!transform.any_op, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)

        //transform.print %tiled_op1 : !transform.any_op

        // Fixme: How come the vector_sizes [1, 64] is not working?
        // transform.structured.vectorize %31 vector_sizes [1, 64] : !transform.any_op
        transform.structured.vectorize %tiled_op1 : !transform.any_op
      }

      transform.yield
    }
  }
}