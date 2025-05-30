#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 4096)>
module {
  func.func @main(%arg0: tensor<8192x8192xf32>, %arg1: tensor<128x8192xf32>) -> tensor<128x8192xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x8192xf32>) -> tensor<128x8192xf32>
    %2 = scf.forall (%arg2, %arg3) in (1, 128) shared_outs(%arg4 = %1) -> (tensor<128x8192xf32>) {
      %c2 = arith.constant 2 : index
      %3 = arith.divui %arg3, %c2 : index
      %4 = arith.remui %arg3, %c2 : index
      %5 = tensor.empty() : tensor<128x128xf32>
      %cst_0 = arith.constant 0.000000e+00 : f32
      %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %7 = affine.apply #map(%3)
      %extracted_slice = tensor.extract_slice %arg4[0, %7] [128, 128] [1, 1] : tensor<128x8192xf32> to tensor<128x128xf32>
      %8 = affine.apply #map1(%4)
      %extracted_slice_1 = tensor.extract_slice %arg1[0, %8] [128, 4096] [1, 1] : tensor<128x8192xf32> to tensor<128x4096xf32>
      %extracted_slice_2 = tensor.extract_slice %arg0[%8, %7] [4096, 128] [1, 1] : tensor<8192x8192xf32> to tensor<4096x128xf32>
      %9 = linalg.matmul ins(%extracted_slice_1, %extracted_slice_2 : tensor<128x4096xf32>, tensor<4096x128xf32>) outs(%6 : tensor<128x128xf32>) -> tensor<128x128xf32>
      %10 = kestrel.dma.reduce %9 into %extracted_slice[0, 0] [1, 1] [128, 128] : tensor<128x128xf32> outs( tensor<128x128xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg4[0, %7] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<128x8192xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %2 : tensor<128x8192xf32>
  }
  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__merge_scf_forall(%arg0: !transform.any_op {transform.consumed}) {
      %0 = transform.kestrel.loop.merge_inner_scf_for_all %arg0 : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.named_sequence @__transform_step2(%arg0: !transform.any_op {transform.readonly}) {
      %0 = transform.param.constant 64 : i64 -> !transform.param<i64>
      %1 = transform.param.constant 64 : i64 -> !transform.param<i64>
      %2 = transform.param.constant 2048 : i64 -> !transform.param<i64>
      %3 = transform.structured.match ops{["scf.forall"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      transform.foreach %3 : !transform.any_op {
      ^bb0(%arg1: !transform.any_op):
        %4 = transform.gpu.map_forall_to_blocks %arg1 generate_gpu_launch : (!transform.any_op) -> !transform.any_op
      }
      transform.yield
    }
    transform.named_sequence @match_func(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["func.func"] : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }
    transform.named_sequence @kestrel_func(%arg0: !transform.any_op {transform.consumed}) {
      %0 = transform.apply_registered_pass "canonicalize" to %arg0 : (!transform.any_op) -> !transform.any_op
      %1 = transform.apply_registered_pass "cse" to %0 : (!transform.any_op) -> !transform.any_op
      %2 = transform.apply_registered_pass "kestrel-convert-linalg-to-aice" to %1 {options = "load-only=0"} : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.named_sequence @kestrel_func_2(%arg0: !transform.any_op {transform.consumed}) {
      %0 = transform.apply_registered_pass "kestrel-post-process-after-bufferization" to %arg0 {options = "load-only=0"} : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) -> !transform.any_op {
      %1 = transform.bufferization.one_shot_bufferize %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op
    }
  }
}