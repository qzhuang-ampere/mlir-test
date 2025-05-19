//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface ----===//
// This file contains the implementation of the BufferizableOpInterface for
// the Kestrel dialect. It provides the necessary methods to support buffer
// analysis and transformation for the Kestrel operations.
//===----------------------------------------------------------------------===//

#include "include/Dialect/Kestrel/TransformOps/BufferizableOpInterfaceImpl.h"
#include "include/Dialect/Kestrel/TransformOps/KestrelOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace kestrel {

struct DMALoadOpInterface
    : public BufferizableOpInterface::ExternalModel<DMALoadOpInterface,
                                                    kestrel::DMALoadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0), BufferRelation::Unknown}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return success();
  }
};

struct DMAStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<DMAStoreOpInterface,
                                                    kestrel::DMAStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0), BufferRelation::Unknown}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return success();
  }
};

}
}

void mlir::kestrel::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, kestrel::KestrelDialect *dialect){
    DMALoadOp::attachInterface<DMALoadOpInterface>(*ctx);
    DMAStoreOp::attachInterface<DMAStoreOpInterface>(*ctx);
  });
}
