//===- BufferizableOpInterfaceImpl.h - Impl. of BufferizableOpInterface ----===//
// This file contains the implementation of the BufferizableOpInterface for
// the Kestrel dialect. It provides the necessary methods to support buffer
// analysis and transformation for the Kestrel operations.
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KESTREL_BUFFERIZABLEOPINTERFACEIMPL_H
#define MLIR_DIALECT_KESTREL_BUFFERIZABLEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace kestrel {
void registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry);
}
}

#endif // MLIR_DIALECT_KESTREL_BUFFERIZABLEOPINTERFACEIMPL_H