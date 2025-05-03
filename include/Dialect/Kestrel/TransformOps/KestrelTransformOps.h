//===- KestrelTransformOps.h - Transform dialect extension -----------------===//
// This file defines a dialect extension header file for the Transform dialect
//===----------------------------------------------------------------------===//

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"

#define GET_OP_CLASSES
#include "KestrelTransformOps.h.inc"

void registerKestrelTransformOps(::mlir::DialectRegistry &registry);
