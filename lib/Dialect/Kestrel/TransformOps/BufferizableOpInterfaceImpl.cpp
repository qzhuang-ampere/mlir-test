//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface ----===//
// This file contains the implementation of the BufferizableOpInterface for
// the Kestrel dialect. It provides the necessary methods to support buffer
// analysis and transformation for the Kestrel operations.
//===----------------------------------------------------------------------===//

#include "include/Dialect/Kestrel/TransformOps/BufferizableOpInterfaceImpl.h"
#include "include/Dialect/Kestrel/TransformOps/KestrelOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace mlir;

namespace mlir {
namespace kestrel {

void registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, kestrel:: *dialect)) {
    registry.attachInterface<DMALoadOpInterface>(*ctx);
    registry.attachInterface<DMAStoreOpInterface>(*ctx);
  }
}

}
}

