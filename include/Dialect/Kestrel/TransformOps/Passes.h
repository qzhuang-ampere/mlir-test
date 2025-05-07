//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
// This file contains the pass entry points for the Kestrel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KESTREL_PASSES_H
#define MLIR_DIALECT_KESTREL_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DECL
#include "include/Dialect/Kestrel/TransformOps/Passes.h.inc"

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "include/Dialect/Kestrel/TransformOps/Passes.h.inc"

} // namespace mlir

#endif // MLIR_DIALECT_KESTREL_PASSES_H