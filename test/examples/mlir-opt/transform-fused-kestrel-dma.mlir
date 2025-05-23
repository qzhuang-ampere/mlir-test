// RUN: kestrel-opt %s --transform-interpreter --split-input-file -canonicalize -cse -kestrel-convert-linalg-to-aice | FileCheck %s
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
    transform.named_sequence @__transform_step1(%arg0: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__transform_step2(%arg0: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @do_nothing(%arg0: !transform.any_op {transform.readonly}) {
      transform.yield
    }

    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      // %0 = transform.foreach_match in %arg0 @__transform_step1 -> @do_nothing
      //       : (!transform.any_op) -> !transform.any_op
      // %1 = transform.foreach_match in %0 @__transform_step2 -> @do_nothing
      //       : (!transform.any_op) -> !transform.any_op
      %csimd_width = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_row = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_col = transform.param.constant 2048 : i64 -> !transform.param<i64>

      %matched_gens = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %gen0, %gen1 = transform.split_handle %matched_gens : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %gen1_tiled, %gen1_forall = transform.structured.tile_using_forall %gen1 tile_sizes [%ct_row, %ct_col] (mapping = [#gpu.block<y>, #gpu.block<x>])
            : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)

      %fuse_0, %loop_0 = transform.structured.fuse_into_containing_op %gen0 into %gen1_forall
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %fuse_1, %loop_1 = transform.structured.fuse_into_containing_op %matmul into %loop_0
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      %fill = transform.structured.match ops{["linalg.fill"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %fuse_2, %loop_2 = transform.structured.fuse_into_containing_op %fill into %loop_1
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

      %gens = transform.structured.match ops{["linalg.generic", "linalg.fill"]} in %loop_2  : (!transform.any_op) -> !transform.any_op
      transform.foreach  %gens : !transform.any_op {
      ^bb1(%gen : !transform.any_op):
        %func, %call = transform.kestrel.loop.outline_with_uniq_name %gen {func_name = "func_Connor"} : (!transform.any_op) -> (!transform.any_op, !transform.op<"func.call">)

        //%vec_target = transform.structured.match ops{["linalg.generic", "linalg.fill"]} in %func : (!transform.any_op) -> !transform.any_op
        //transform.structured.vectorize %func vector_sizes[64, 64] : !transform.any_op
        //%veced_tile = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op
      }
      transform.yield
    }
  }

}