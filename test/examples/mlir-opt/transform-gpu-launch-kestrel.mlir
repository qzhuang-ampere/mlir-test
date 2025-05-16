// RUN: mlir-opt %s -one-shot-bufferize="copy-before-write unknown-type-conversion=identity-layout-map" --transform-interpreter --split-input-file -canonicalize -cse | FileCheck %s
// -linalg-block-pack-matmul="block-factors=32,16,64 lhs-transpose-outer-blocks=false lhs-transpose-inner-blocks=false rhs-transpose-outer-blocks=true rhs-transpose-inner-blocks=true" -canonicalize

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

module {
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<1x4096xf32>, %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2 = linalg.matmul ins(%arg2, %arg0 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%1 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>

    return %2 : tensor<4096x4096xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %csimd_width = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_row = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_col = transform.param.constant 2048 : i64 -> !transform.param<i64>

      %mat = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      //transform.print %mat : !transform.any_op

      %tiled_op_mat, %forall_op_mat = transform.structured.tile_using_forall %mat tile_sizes [%ct_row, %ct_col] (mapping = [#gpu.block<y>, #gpu.block<x>])
           : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)

      transform.print %tiled_op_mat : !transform.any_op
      transform.print %forall_op_mat : !transform.any_op
      //transform.print %gpu_launch0 : !transform.any_op
      //transform.print %gens : !transform.any_op

      transform.yield
    }
  }
}