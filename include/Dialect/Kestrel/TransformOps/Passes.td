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
  let options =
      [Option<"loadOnly", "load-only",
              "bool", /*default=*/"false",
              "Load the pass only, will use it later, work around for pass used by transform dialect.">,
  ];
}

def PostProcessAfterBufferization : Pass<"kestrel-post-process-after-bufferization"> {
  let summary = "Post processing after bufferization";
  let description = [{
    This pass performs post-processing after bufferization.
    1. Lower parallel insert_slice dma.store
    2. Deal with gemm size
    3. Add dependencies
  }];
  let dependentDialects = ["memref::MemRefDialect", "kestrel::KestrelDialect"];
  let options =
      [Option<"loadOnly", "load-only",
              "bool", /*default=*/"false",
              "Load the pass only, will use it later, work around for pass used by transform dialect.">,
  ];
}
#endif // MLIR_DIALECT_KESTREL_PASSES_TD