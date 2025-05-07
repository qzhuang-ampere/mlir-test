//===-- Passes.td - Kestrel pass definition file ------------*- tablegen -*-===//
// This file defines the passes for the Kestrel dialect.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KESTREL_PASSES_TD
#define MLIR_DIALECT_KESTREL_PASSES_TD
include "mlir/Pass/PassBase.td"

def ConvertLinalgToAice : Pass<"kestrel-convert-linalg-to-aice"> {
  let summary = "Convert linalg ops to aice ops";
  let description = [{
    This pass converts linalg ops to aice ops.
  }];
  let dependentDialects = ["linalg::LinalgDialect", "kestrel::KestrelDialect"];
}

#endif // MLIR_DIALECT_KESTREL_PASSES_TD