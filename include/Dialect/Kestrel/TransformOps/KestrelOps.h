//===- KestrelOps.h - Kestrel dialect ops ---------------------------------===//
// This is the operation definition header file for the Kestrel dialect.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KESTREL_IR_KESTRELOPS_H
#define MLIR_DIALECT_KESTREL_IR_KESTRELOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "KestrelDialect.h.inc"

#define GET_OP_CLASSES
#include "KestrelOps.h.inc"

#endif // MLIR_DIALECT_KESTREL_IR_KESTRELOPS_H