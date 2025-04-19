// Tiling: mlir-opt ./gemm_relu_tiled.mlir -transform-interpreter -cse --one-shot-bufferize="copy-before-write unknown-type-conversion=identity-layout-map" -convert-linalg-to-loops -canonicalize
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0, d1) -> (0, d1)>
module {
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<1x4096xf32>, %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = tensor.empty() : tensor<128x64x32x64xf32>
    %pack = linalg.pack %arg2 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 64] into %1 : tensor<4096x4096xf32> -> tensor<128x64x32x64xf32>
    %2 = tensor.empty() : tensor<256x64x16x64xf32>
    %pack_0 = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [1, 0] inner_tiles = [16, 64] into %2 : tensor<4096x4096xf32> -> tensor<256x64x16x64xf32>
    %3 = tensor.empty() : tensor<128x256x32x16xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<128x256x32x16xf32>) -> tensor<128x256x32x16xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<128x64x32x64xf32>, tensor<256x64x16x64xf32>) outs(%4 : tensor<128x256x32x16xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %8 = arith.mulf %in, %in_1 : f32
      %9 = arith.addf %out, %8 : f32
      linalg.yield %9 : f32
    } -> tensor<128x256x32x16xf32>
    %unpack = linalg.unpack %5 inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %0 : tensor<128x256x32x16xf32> -> tensor<4096x4096xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel"]} ins(%unpack, %arg1 : tensor<4096x4096xf32>, tensor<1x4096xf32>) outs(%0 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %8 = arith.addf %in, %in_1 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x4096xf32>
    %7 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<4096x4096xf32>) outs(%0 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.cmpf ugt, %in, %cst : f32
      %9 = arith.select %8, %in, %cst : f32
      linalg.yield %9 : f32
    } -> tensor<4096x4096xf32>
    return %7 : tensor<4096x4096xf32>
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op {transform.readonly}) {
    %pack = transform.structured.match ops{["linalg.pack"]} in %module_op
      : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %pack : (!transform.op<"linalg.pack">)
      -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
      transform.yield
  }
}