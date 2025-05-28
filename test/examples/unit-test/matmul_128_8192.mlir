// RUN: kestrel-opt --transform-interpreter --split-input-file -canonicalize -cse %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<8192x8192xf32>, %arg1: tensor<128x8192xf32>) -> tensor<128x8192xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x8192xf32>) -> tensor<128x8192xf32>
    %2 = linalg.matmul ins(%arg1, %arg0 : tensor<128x8192xf32>, tensor<8192x8192xf32>) outs(%1 : tensor<128x8192xf32>) -> tensor<128x8192xf32>
    return %2 : tensor<128x8192xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_step1(%arg0: !transform.any_op {transform.readonly}) {
      %ct_row = transform.param.constant 128 : i64 -> !transform.param<i64>
      %ct_col = transform.param.constant 128 : i64 -> !transform.param<i64>

      %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %fuse_1, %loop_1 = transform.structured.tile_using_forall %matmul tile_sizes [%ct_row, %ct_col] (mapping = [#gpu.block<y>, #gpu.block<x>])
            : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)

      %1, %2, %3, %loop = transform.structured.tile_reduction_using_forall %fuse_1
        by num_threads = [0, 0, 2], tile_sizes = [], mapping = [#gpu.block<x>] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

      transform.yield
    }

    transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
      transform.include @__transform_step1 failures(propagate) (%arg0) : (!transform.any_op) -> ()
      transform.yield
    }
  }
}