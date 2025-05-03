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

#endif // KESTREL_OPS
