//===- KestrelOps.cpp - Kestrel dialect ops --------------------------------===//
// This is the operation definition file for the Kestrel dialect.
//===----------------------------------------------------------------------===//

#include "KestrelOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

#include "KestrelDialect.cpp.inc"

void ::mlir::kestrel::KestrelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "KestrelOps.cpp.inc"
      >();
declarePromisedInterfaces<
    bufferization::BufferizableOpInterface, DMALoadOp, DMAStoreWithResultOp, AiceMatMulOp>();
}

#define GET_OP_CLASSES
#include "KestrelOps.cpp.inc"