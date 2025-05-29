// RUN: kestrel-opt -kestrel-convert-linalg-to-aice="load-only" --kestrel-post-process-after-bufferization="load-only" --transform-interpreter --split-input-file -canonicalize -cse %s | FileCheck %s
#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 4096)>
module {
  func.func @main(%arg0: tensor<8192x8192xf32>, %arg1: tensor<128x8192xf32>) -> tensor<128x8192xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x8192xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x8192xf32>) -> tensor<128x8192xf32>
    %2 = scf.forall (%arg2, %arg3) in (1, 64) shared_outs(%arg4 = %1) -> (tensor<128x8192xf32>) {
      %3 = affine.apply #map(%arg3)
      %extracted_slice = tensor.extract_slice %arg0[0, %3] [8192, 128] [1, 1] : tensor<8192x8192xf32> to tensor<8192x128xf32>
      %extracted_slice_0 = tensor.extract_slice %arg4[0, %3] [128, 128] [1, 1] : tensor<128x8192xf32> to tensor<128x128xf32>
      %4 = tensor.empty() : tensor<128x128x2xf32>
      %5 = linalg.fill ins(%cst : f32) outs(%4 : tensor<128x128x2xf32>) -> tensor<128x128x2xf32>
      %6 = scf.forall (%arg5) in (2) shared_outs(%arg6 = %5) -> (tensor<128x128x2xf32>) {
        %extracted_slice_1 = tensor.extract_slice %arg6[0, 0, %arg5] [128, 128, 1] [1, 1, 1] : tensor<128x128x2xf32> to tensor<128x128xf32>
        %7 = affine.apply #map1(%arg5)
        %extracted_slice_2 = tensor.extract_slice %arg1[0, %7] [128, 4096] [1, 1] : tensor<128x8192xf32> to tensor<128x4096xf32>
        %extracted_slice_3 = tensor.extract_slice %extracted_slice[%7, 0] [4096, 128] [1, 1] : tensor<8192x128xf32> to tensor<4096x128xf32>
        %8 = linalg.matmul ins(%extracted_slice_2, %extracted_slice_3 : tensor<128x4096xf32>, tensor<4096x128xf32>) outs(%extracted_slice_1 : tensor<128x128xf32>) -> tensor<128x128xf32>
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %8 into %arg6[0, 0, %arg5] [128, 128, 1] [1, 1, 1] : tensor<128x128xf32> into tensor<128x128x2xf32>
        }
      } {mapping = [#gpu.block<x>]}
      %reduced = linalg.reduce ins(%6 : tensor<128x128x2xf32>) outs(%extracted_slice_0 : tensor<128x128xf32>) dimensions = [2]
        (%in: f32, %init: f32) {
          %7 = arith.addf %in, %init : f32
          linalg.yield %7 : f32
        }
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %reduced into %arg4[0, %3] [128, 128] [1, 1] : tensor<128x128xf32> into tensor<128x8192xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %2 : tensor<128x8192xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__merge_scf_forall(%arg0: !transform.any_op {transform.consumed}) {
      %1 = transform.kestrel.loop.merge_inner_scf_for_all %arg0 : (!transform.any_op) -> !transform.any_op
      transform.yield
    }

    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.consumed}) {
      %0 = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op

      transform.foreach  %0 : !transform.any_op {
      ^bb1(%each : !transform.any_op):
        transform.include @__merge_scf_forall failures(suppress) (%each) : (!transform.any_op) -> ()
      }

      %1 = transform.bufferization.one_shot_bufferize %arg1 { bufferize_function_boundaries = true }
           : (!transform.any_op) -> !transform.any_op

      transform.yield
    }
  }
}