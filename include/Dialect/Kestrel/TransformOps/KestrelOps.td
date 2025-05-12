//===- KestrelOps.td - Kestrel dialect ops ---------------------------------===//
// This is the operation definition file for the Kestrel dialect.
//===----------------------------------------------------------------------===//

#ifndef KESTREL_OPS
#define KESTREL_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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

def AiceMatMulOp : KestrelOp<"aice_matmul"> {
  let summary = "AICE matrix multiplication operation";
  let description = [{
    This operation performs matrix multiplication on two input matrices
    and produces an output matrix.
  }];

  let arguments = (ins AnyTensor : $lhs,
                       AnyTensor : $rhs);
  let results = (outs AnyTensor : $result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) type($rhs) `->` type($result)";
}

def DMALoadOp : KestrelOp<"dma.load"> {
  let summary = "DMA load operation";
  let description = [{
    This operation performs a DMA load from memory into a specified buffer.
  }];

  let arguments = (ins AnyTensor : $buffer,
                       I32 : $offset,
                       I32 : $stride,
                       I32 : $size);
  let results = (outs AnyTensor : $result);

  let assemblyFormat = [{
    `dma.load` $buffer attr-dict `offset` $offset `stride` $stride `size` $size `:`
     type($buffer) type($offset) type($stride) type($size) `->` type($result)
    }];
}

def DMAStoreOp : KestrelOp<"dma.store"> {
  let summary = "DMA store operation";
  let description = [{
    This operation performs a DMA store from a specified buffer to memory.
  }];

  let arguments = (ins AnyTensor : $buffer,
                       I32 : $offset,
                       I32 : $stride,
                       I32 : $size);
  let results = (outs AnyTensor : $result);

  let assemblyFormat = [{
    `dma.store` $buffer attr-dict `offset` $offset `stride` $stride `size` $size `:`
     type($buffer) type($offset) type($stride) type($size) `->` type($result)
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
