// RUN: mlir-opt %s --transform-interpreter --split-input-file -canonicalize -cse --one-shot-bufferize="copy-before-write unknown-type-conversion=identity-layout-map" -canonicalize -cse  | FileCheck %s

// This is a simple tile-and-fuse example with a single fusion group.

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (0, d1)>

module {
  // CHECK: func @foo
  // CHECK:   scf.forall {{.*}} {
  // CHECK:     linalg.fill
  // CHECK:     linalg.matmul
  // CHECK:     linalg.generic
  // CHECK:   }

  // func.func @foo(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?xf32>,
  //                %D: tensor<?x?xf32>, %sz0: index, %sz1: index)
  //     -> tensor<?x?xf32>
  // {
  //   %cst = arith.constant 0.000000e+00 : f32
  //   %5 = linalg.fill
  //       {__producer__}
  //       ins(%cst : f32)
  //       outs(%D : tensor<?x?xf32>) -> tensor<?x?xf32>
  //   %6 = linalg.matmul
  //       {__producer__}
  //       ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
  //       outs(%5 : tensor<?x?xf32>) -> tensor<?x?xf32>
  //   %7 = linalg.generic
  //       {__root__,
  //        indexing_maps = [affine_map<(d0, d1) -> (d0)>,
  //                         affine_map<(d0, d1) -> (d0, d1)>,
  //                         affine_map<(d0, d1) -> (d0, d1)>],
  //        iterator_types = ["parallel", "parallel"]
  //       }
  //       ins(%C, %6 : tensor<?xf32>, tensor<?x?xf32>)
  //       outs(%D : tensor<?x?xf32>) {
  //   ^bb0(%arg2: f32, %arg3: f32, %arg4: f32):
  //     %16 = arith.maximumf %arg3, %cst : f32
  //     %17 = arith.cmpf ogt, %arg2, %cst : f32
  //     %18 = arith.select %17, %cst, %16 : f32
  //     linalg.yield %18 : f32
  //   } -> tensor<?x?xf32>
  //   return %7 : tensor<?x?xf32>
  // }

  // Bufferize: mlir-opt ./gemm_relu.mlir --one-shot-bufferize="copy-before-write unknown-type-conversion=identity-layout-map" -cse
  // Tiling: mlir-opt ./gemm_relu.mlir -linalg-block-pack-matmul=block-factors=32,16,64 -canonicalize


  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<1x4096xf32>, %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.fill {__producer__} ins(%cst : f32) outs(%0 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %2 = linalg.matmul {__producer__} ins(%arg2, %arg0 : tensor<4096x4096xf32>, tensor<4096x4096xf32>) outs(%1 : tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    %3 = linalg.generic {__producer__, indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg1 : tensor<4096x4096xf32>, tensor<1x4096xf32>) outs(%0 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %5 = arith.addf %in, %in_0 : f32
      linalg.yield %5 : f32
    } -> tensor<4096x4096xf32>
    %4 = linalg.generic {__root__, indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<4096x4096xf32>) outs(%0 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.cmpf ugt, %in, %cst : f32
      %6 = arith.select %5, %in, %cst : f32
      linalg.yield %6 : f32
    } -> tensor<4096x4096xf32>
    return %4 : tensor<4096x4096xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      // Find the root and all producers.
      %root = transform.structured.match attributes{"__root__"} in %arg1 : (!transform.any_op) -> !transform.any_op
      %producers = transform.structured.match attributes{"__producer__"} in %arg1 : (!transform.any_op) -> !transform.any_op

      // Tile the root.
      %tiled_op, %forall_op = transform.structured.tile_using_forall %root num_threads [8, 4]
           : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      // Fuse all producers.
      transform.structured.fuse_into_containing_op %producers into %forall_op
        : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.yield
    }
  }
}