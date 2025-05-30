//===- KestrelOps.td - Kestrel dialect ops ---------------------------------===//
// This is the operation definition file for the Kestrel dialect.
//===----------------------------------------------------------------------===//

#ifndef KESTREL_OPS
#define KESTREL_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/EnumAttr.td"

def KestrelDialect : Dialect {
  let name = "kestrel";
  let cppNamespace = "::mlir::kestrel";
  let description = [{
    The Kestrel dialect is a custom dialect for the Kestrel compiler.
    It is used to represent high-level operations and transformations
    that are specific to the Kestrel compiler.
  }];
}

class KestrelOp<string mnemonic, list<Trait> traits = []> :
    Op<KestrelDialect, mnemonic, traits>;

def Kestrel_ReductionKind : I32EnumAttr<"ReductionKind",
  "Reduction of an iterator/mesh dimension.", [
  I32EnumAttrCase<"Sum", 1, "sum">,
  I32EnumAttrCase<"Max", 2, "max">,
  I32EnumAttrCase<"Min", 3, "min">,
  I32EnumAttrCase<"Product", 4, "product">,
  I32EnumAttrCase<"Average", 5, "average">,
  I32EnumAttrCase<"BitwiseAnd", 6, "bitwise_and">,
  I32EnumAttrCase<"BitwiseOr", 7, "bitwise_or">,
  I32EnumAttrCase<"BitwiseXor", 8, "bitwise_xor">,
  I32EnumAttrCase<"Generic", 100, "generic">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::kestrel";
}

def Kestrel_ReductionKindAttr : EnumAttr<KestrelDialect, Kestrel_ReductionKind, "partial"> {
  let assemblyFormat = "$value";
}

def Kestrel_MeshAxesAttr : DenseArrayAttrBase<"DenseI16ArrayAttr", "int16_t", "i16">;

def AiceMatMulOp : KestrelOp<"aice_matmul"> {
  let summary = "AICE matrix multiplication operation";
  let description = [{
    This operation performs matrix multiplication on two input matrices
    and produces an output matrix.
  }];

  let arguments = (ins AnyType : $lhs,
                       AnyType : $rhs,
                       AnyType : $out);
  let results = (outs AnyType : $result);

  let assemblyFormat = "$lhs `,` $rhs `,` $out attr-dict `:` type($lhs) type($rhs) type($out) `->` type($result)";
}

def DMALoadOp : KestrelOp<"dma.load", [
    AttrSizedOperandSegments]> {
  let summary = "DMA load operation";
  let description = [{
    This operation performs a DMA load from memory into a specified buffer.
  }];

  let arguments = (ins AnyType:$source,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyType : $result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($result)
    }];
}

def DMAParallelStoreOp : KestrelOp<"dma.parallel_store", [
    AttrSizedOperandSegments]> {
  let summary = "DMA store operation";
  let description = [{
    This operation performs a DMA store from a specified buffer to memory.
  }];

  let arguments = (ins AnyType : $source,
                       AnyType : $dest,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);

  let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($dest)
  }];

}

def DMAStoreWithResultOp : KestrelOp<"dma.store_with_result", [
    AttrSizedOperandSegments,
    TypesMatchWith<"expected result type to match dest type",
                   "dest", "result", "$_self">]> {
  let summary = "DMA store operation";
  let description = [{
    This operation performs a DMA store from a specified buffer to memory.
  }];

  let arguments = (ins AnyType : $source,
                       AnyType : $dest,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs AnyType : $result);

  let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($dest)
  }];
}

def DMAStoreOp : KestrelOp<"dma.store", [
    AttrSizedOperandSegments]> {
  let summary = "DMA store operation";
  let description = [{
    This operation performs a DMA store from a specified buffer to memory.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the source memref",
                        [MemReadAt<0, FullEffect>]>:$source,
                       Arg<AnyRankedOrUnrankedMemRef, "the target memref",
                         [MemWriteAt<1, FullEffect>]>:$dest,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);

  let assemblyFormat = [{
    $source `into` $dest ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($dest)
  }];
}

def DMAReduceOp : KestrelOp<"dma.reduce", [
    DestinationStyleOpInterface,
    AttrSizedOperandSegments]> {
  let summary = "DMA reduce operation";
  let description = [{
    This operation performs a DMA reduction from a specified buffer to memory.
  }];

  let arguments = (ins AnyType : $source,
                       AnyType : $dest,
                       Variadic<Index>:$offsets,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$strides,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$static_sizes,
                       DenseI64ArrayAttr:$static_strides);
  let results = (outs Variadic<AnyType> : $results);

  let assemblyFormat = [{
    $source `into` $dest
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` type($source) `to` type($dest) `->` type($results)
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() { return getDestMutable(); }
  }];
}

def Kestrel_AllReduceOp : KestrelOp<"all_reduce", [
    Pure,
    SameOperandsAndResultShape]> {
  let summary = "All-reduce operation";
  let description = [{
    This operation performs an all-reduce operation on the input tensor
    across a device mesh.
  }];

  let arguments = (ins AnyType : $input,
                       DefaultValuedAttr<Kestrel_MeshAxesAttr, "{}">: $mesh_axes,
                       DefaultValuedAttr<Kestrel_ReductionKindAttr, "mlir::kestrel::ReductionKind::Sum"> : $reduction);
  let results = (outs AnyType : $result);

  let assemblyFormat = [{
    `all_reduce` $input `reduction` $reduction attr-dict `:`
     type($input) `->` type($result)
  }];
}

def SyncOp : KestrelOp<"sync"> {
  let summary = "Synchronization operation";
  let description = [{
    This operation synchronizes of both kernel executions and memory operations.
  }];

  let arguments = (ins I32Attr : $sync_scope);
  let results = (outs);

  let assemblyFormat = "`sync` attr-dict";
}

def AiceMatMulWithDepOp : KestrelOp<"aice_matmul_with_dep"> {
  let summary = "AICE matrix multiplication operation with dependencies";
  let description = [{
    This operation performs matrix multiplication on two input matrices
    and produces an output matrix, with dependencies on other operations.
  }];

  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs,
                       I32 : $doid0,
                       I32 : $doid1,
                       I32 : $doid2,
                       I32 : $did);
  let results = (outs AnyTensor : $result);

  let assemblyFormat = [{
    `aice_matmul_with_dep` $lhs `,` $rhs `,` `doids` $doid0 `,` $doid1 `,` $doid2 `did` $did attr-dict `:`
     type($lhs) type($rhs) type($doid0) type($doid1) type($doid2) type($did) `->` type($result)
  }];
}

def DMALoadWithDepOp : KestrelOp<"dma.load_with_dep"> {
  let summary = "DMA load operation with dependencies";
  let description = [{
    This operation performs a DMA load from memory into a specified buffer,
    with dependencies on other operations.
  }];

  let arguments = (ins AnyTensor : $buffer,
                       I32 : $offset,
                       I32 : $stride,
                       I32 : $size,
                       I32 : $doid0,
                       I32 : $doid1,
                       I32 : $doid2,
                       I32 : $did);
  let results = (outs AnyTensor : $result);

  let assemblyFormat = [{
    `dma.load_with_dep` $buffer `offset` $offset `stride` $stride `size` $size
     `,` `doids` $doid0 `,` $doid1 `,` $doid2 `did` $did attr-dict `:`
     type($buffer) type($offset) type($stride) type($size) type($doid0) type($doid1) type($doid2) type($did)
     `->` type($result)
  }];
}

def DMAStoreWithDepOp : KestrelOp<"dma.store_with_dep"> {
  let summary = "DMA store operation with dependencies";
  let description = [{
    This operation performs a DMA store from a specified buffer to memory,
    with dependencies on other operations.
  }];

  let arguments = (ins AnyTensor : $buffer,
                       I32 : $offset,
                       I32 : $stride,
                       I32 : $size,
                       I32 : $doid0,
                       I32 : $doid1,
                       I32 : $doid2,
                       I32 : $did);
  let results = (outs AnyTensor : $result);

  let assemblyFormat = [{
    `dma.store_with_dep` $buffer `offset` $offset `stride` $stride `size` $size
     `,` `doids` $doid0 `,` $doid1 `,` $doid2 `did` $did attr-dict `:`
     type($buffer) type($offset) type($stride) type($size) type($doid0) type($doid1) type($doid2) type($did)
     `->` type($result)
  }];
}
#endif // KESTREL_OPS
