// RUN: mlir-opt %s -one-shot-bufferize="copy-before-write bufferize-function-boundaries unknown-type-conversion=identity-layout-map" --transform-interpreter --split-input-file -canonicalize -cse | FileCheck %s
// -linalg-block-pack-matmul="block-factors=32,16,64 lhs-transpose-outer-blocks=false lhs-transpose-inner-blocks=false rhs-transpose-outer-blocks=true rhs-transpose-inner-blocks=true" -canonicalize
#map = affine_map<(d0, d1) -> (0, d1)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<(d0) -> (d0 * 2048)>
module {
  func.func @func_Connor_Nd5xcSG9(%arg0: f32, %arg1: index, %arg2: index, %arg3: index, %arg4: tensor<64x2048xf32>) -> tensor<64x2048xf32> {
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg5 = %c0 to %c2048 step %c64 iter_args(%arg6 = %arg4) -> (tensor<64x2048xf32>) {
      %extracted_slice = tensor.extract_slice %arg6[0, %arg5] [64, 64] [1, 1] : tensor<64x2048xf32> to tensor<64x64xf32>
      %1 = vector.broadcast %arg0 : f32 to vector<64x64xf32>
      %2 = vector.transfer_write %1, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, tensor<64x64xf32>
      %inserted_slice = tensor.insert_slice %2 into %arg6[0, %arg5] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<64x2048xf32>
      scf.yield %inserted_slice : tensor<64x2048xf32>
    }
    return %0 : tensor<64x2048xf32>
  }
  func.func @func_Connor_s07ilRjW(%arg0: tensor<64x2048xf32>, %arg1: tensor<1x2048xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: tensor<64x2048xf32>) -> tensor<64x2048xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg6 = %c0 to %c2048 step %c64 iter_args(%arg7 = %arg5) -> (tensor<64x2048xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[0, %arg6] [64, 64] [1, 1] : tensor<64x2048xf32> to tensor<64x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[0, %arg6] [1, 64] [1, 1] : tensor<1x2048xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg7[0, %arg6] [64, 64] [1, 1] : tensor<64x2048xf32> to tensor<64x64xf32>
      %1 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x64xf32>
      %2 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true], permutation_map = #map} : tensor<1x64xf32>, vector<64x64xf32>
      %3 = arith.addf %1, %2 : vector<64x64xf32>
      %4 = vector.transfer_write %3, %extracted_slice_1[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, tensor<64x64xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg7[0, %arg6] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<64x2048xf32>
      scf.yield %inserted_slice : tensor<64x2048xf32>
    }
    return %0 : tensor<64x2048xf32>
  }
  func.func @func_Connor_6vhieePh(%arg0: tensor<64x2048xf32>, %arg1: f32, %arg2: index, %arg3: index, %arg4: index, %arg5: tensor<64x2048xf32>) -> tensor<64x2048xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg6 = %c0 to %c2048 step %c64 iter_args(%arg7 = %arg5) -> (tensor<64x2048xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[0, %arg6] [64, 64] [1, 1] : tensor<64x2048xf32> to tensor<64x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg7[0, %arg6] [64, 64] [1, 1] : tensor<64x2048xf32> to tensor<64x64xf32>
      %1 = vector.transfer_read %extracted_slice[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<64x64xf32>, vector<64x64xf32>
      %2 = vector.broadcast %arg1 : f32 to vector<64x64xf32>
      %3 = arith.cmpf ugt, %1, %2 : vector<64x64xf32>
      %4 = arith.select %3, %1, %2 : vector<64x64xi1>, vector<64x64xf32>
      %5 = vector.transfer_write %4, %extracted_slice_0[%c0, %c0] {in_bounds = [true, true]} : vector<64x64xf32>, tensor<64x64xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg7[0, %arg6] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<64x2048xf32>
      scf.yield %inserted_slice : tensor<64x2048xf32>
    }
    return %0 : tensor<64x2048xf32>
  }
  func.func @func_Aice_YjkRTEOE(%arg0: tensor<64x4096xf32>, %arg1: tensor<4096x2048xf32>, %arg2: tensor<64x2048xf32>) -> tensor<64x2048xf32> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %0 = scf.for %arg3 = %c0 to %c2048 step %c64 iter_args(%arg4 = %arg2) -> (tensor<64x2048xf32>) {
      %extracted_slice = tensor.extract_slice %arg1[0, %arg3] [4096, 64] [1, 1] : tensor<4096x2048xf32> to tensor<4096x64xf32>
      %extracted_slice_0 = tensor.extract_slice %arg4[0, %arg3] [64, 64] [1, 1] : tensor<64x2048xf32> to tensor<64x64xf32>
      %1 = kestrel.aice_matmul %arg0, %extracted_slice : tensor<64x4096xf32> tensor<4096x64xf32> -> tensor<64x64xf32>
      %inserted_slice = tensor.insert_slice %1 into %arg4[0, %arg3] [64, 64] [1, 1] : tensor<64x64xf32> into tensor<64x2048xf32>
      scf.yield %inserted_slice : tensor<64x2048xf32>
    }
    return %0 : tensor<64x2048xf32>
  }
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<1x4096xf32>, %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = scf.forall (%arg3, %arg4) in (64, 2) shared_outs(%arg5 = %0) -> (tensor<4096x4096xf32>) {
      %5 = affine.apply #map1(%arg3)
      %6 = affine.apply #map2(%arg4)
      %7 = kestrel.dma.load %arg5[%5, %6] [1, 1] [64, 2048] : tensor<4096x4096xf32> to tensor<64x2048xf32>
      %8 = func.call @func_Connor_Nd5xcSG9(%cst, %c0, %c2048, %c64, %7) : (f32, index, index, index, tensor<64x2048xf32>) -> tensor<64x2048xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %8 into %arg5[%5, %6] [64, 2048] [1, 1] : tensor<64x2048xf32> into tensor<4096x4096xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %2 = scf.forall (%arg3, %arg4) in (64, 2) shared_outs(%arg5 = %1) -> (tensor<4096x4096xf32>) {
      %5 = affine.apply #map1(%arg3)
      %6 = affine.apply #map2(%arg4)
      %7 = kestrel.dma.load %arg2[%5, 0] [1, 1] [64, 4096] : tensor<4096x4096xf32> to tensor<64x4096xf32>
      %8 = kestrel.dma.load %arg0[0, %6] [1, 1] [4096, 2048] : tensor<4096x4096xf32> to tensor<4096x2048xf32>
      %9 = kestrel.dma.load %arg5[%5, %6] [1, 1] [64, 2048] : tensor<4096x4096xf32> to tensor<64x2048xf32>
      %10 = func.call @func_Aice_YjkRTEOE(%7, %8, %9) : (tensor<64x4096xf32>, tensor<4096x2048xf32>, tensor<64x2048xf32>) -> tensor<64x2048xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg5[%5, %6] [64, 2048] [1, 1] : tensor<64x2048xf32> into tensor<4096x4096xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %3 = scf.forall (%arg3, %arg4) in (64, 2) shared_outs(%arg5 = %0) -> (tensor<4096x4096xf32>) {
      %5 = affine.apply #map1(%arg3)
      %6 = affine.apply #map2(%arg4)
      %7 = kestrel.dma.load %2[%5, %6] [1, 1] [64, 2048] : tensor<4096x4096xf32> to tensor<64x2048xf32>
      %8 = kestrel.dma.load %arg1[0, %6] [1, 1] [1, 2048] : tensor<1x4096xf32> to tensor<1x2048xf32>
      %9 = kestrel.dma.load %arg5[%5, %6] [1, 1] [64, 2048] : tensor<4096x4096xf32> to tensor<64x2048xf32>
      %10 = func.call @func_Connor_s07ilRjW(%7, %8, %c0, %c2048, %c64, %9) : (tensor<64x2048xf32>, tensor<1x2048xf32>, index, index, index, tensor<64x2048xf32>) -> tensor<64x2048xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %10 into %arg5[%5, %6] [64, 2048] [1, 1] : tensor<64x2048xf32> into tensor<4096x4096xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %4 = scf.forall (%arg3, %arg4) in (64, 2) shared_outs(%arg5 = %0) -> (tensor<4096x4096xf32>) {
      %5 = affine.apply #map1(%arg3)
      %6 = affine.apply #map2(%arg4)
      %7 = kestrel.dma.load %3[%5, %6] [1, 1] [64, 2048] : tensor<4096x4096xf32> to tensor<64x2048xf32>
      %8 = kestrel.dma.load %arg5[%5, %6] [1, 1] [64, 2048] : tensor<4096x4096xf32> to tensor<64x2048xf32>
      %9 = func.call @func_Connor_6vhieePh(%7, %cst, %c0, %c2048, %c64, %8) : (tensor<64x2048xf32>, f32, index, index, index, tensor<64x2048xf32>) -> tensor<64x2048xf32>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %9 into %arg5[%5, %6] [64, 2048] [1, 1] : tensor<64x2048xf32> into tensor<4096x4096xf32>
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return %4 : tensor<4096x4096xf32>
  }

  module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
      %csimd_width = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_row = transform.param.constant 64 : i64 -> !transform.param<i64>
      %ct_col = transform.param.constant 2048 : i64 -> !transform.param<i64>

      %for_alls = transform.structured.match ops{["scf.forall"]} in %arg1 : (!transform.any_op) -> !transform.any_op
      //transform.print %for_alls : !transform.any_op
      //transform.foreach  %for_alls : !transform.any_op {
      ^bb1(%for_all : !transform.any_op):
        %gpuLaunch = transform.gpu.map_forall_to_blocks %for_all { generate_gpu_launch } : (!transform.any_op) -> !transform.any_op
      }

      // %tiled_op_mat, %forall_op_mat = transform.structured.tile_using_forall %mat tile_sizes [%ct_row, %ct_col] (mapping = [#gpu.block<y>, #gpu.block<x>])
      //      : (!transform.any_op, !transform.param<i64>, !transform.param<i64>) -> (!transform.any_op, !transform.any_op)

      //transform.print %tiled_op_mat : !transform.any_op
      //transform.print %forall_op_mat : !transform.any_op
      //transform.print %gpu_launch0 : !transform.any_op
      //transform.print %gens : !transform.any_op

      transform.yield
    }
  }
}