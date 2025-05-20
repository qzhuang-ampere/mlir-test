//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface ----===//
// This file contains the implementation of the BufferizableOpInterface for
// the Kestrel dialect. It provides the necessary methods to support buffer
// analysis and transformation for the Kestrel operations.
//===----------------------------------------------------------------------===//

#include "include/Dialect/Kestrel/TransformOps/BufferizableOpInterfaceImpl.h"
#include "include/Dialect/Kestrel/TransformOps/KestrelOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {
namespace kestrel {

struct DMALoadOpInterface
    : public BufferizableOpInterface::ExternalModel<DMALoadOpInterface,
                                                    kestrel::DMALoadOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dmaLoadOp = cast<kestrel::DMALoadOp>(op);
    FailureOr<Value> memref =
        getBuffer(rewriter, dmaLoadOp.getSource(), options);
    if (failed(memref))
      return failure();

    auto resultMemrefType =
        bufferization::getBufferType(dmaLoadOp.getResult(), options);
    if (failed(resultMemrefType))
      return failure();

    auto newOp = rewriter.create<kestrel::DMALoadOp>(
        dmaLoadOp->getLoc(), llvm::cast<MemRefType>(*resultMemrefType),
        *memref,
        dmaLoadOp.getOffsets(),
        dmaLoadOp.getStrides(),
        dmaLoadOp.getSizes(),
        dmaLoadOp.getStaticOffsets(),
        dmaLoadOp.getStaticStrides(),
        dmaLoadOp.getStaticSizes());

    replaceOpWithBufferizedValues(rewriter, op, newOp.getResult());
    return success();
  }
};

struct DMAStoreOpInterface
    : public BufferizableOpInterface::ExternalModel<DMAStoreOpInterface,
                                                    kestrel::DMAStoreOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dmaStoreOp = cast<kestrel::DMAStoreOp>(op);
    FailureOr<Value> srcMemref =
        getBuffer(rewriter, dmaStoreOp.getSource(), options);
    if (failed(srcMemref))
      return failure();

    FailureOr<Value> dstMemref =
        getBuffer(rewriter, dmaStoreOp.getDest(), options);
    if (failed(dstMemref))
      return failure();

    auto resultMemrefType =
        bufferization::getBufferType(dmaStoreOp.getResult(), options);
    if (failed(resultMemrefType))
      return failure();

    auto newOp = rewriter.create<kestrel::DMAStoreOp>(
        dmaStoreOp->getLoc(), llvm::cast<MemRefType>(*resultMemrefType),
        *srcMemref,
        *dstMemref,
        dmaStoreOp.getOffsets(),
        dmaStoreOp.getStrides(),
        dmaStoreOp.getSizes(),
        dmaStoreOp.getStaticOffsets(),
        dmaStoreOp.getStaticStrides(),
        dmaStoreOp.getStaticSizes());

    replaceOpWithBufferizedValues(rewriter, op, newOp.getResult());
    return success();
  }
};

struct AiceMatMulOpInterface
    : public BufferizableOpInterface::ExternalModel<AiceMatMulOpInterface,
                                                    kestrel::AiceMatMulOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    llvm::dbgs() << "Bufferizing AiceMatMulOp\n";
    auto aiceMatMulOp = cast<kestrel::AiceMatMulOp>(op);
    FailureOr<Value> lhsMemref =
        getBuffer(rewriter, aiceMatMulOp.getLhs(), options);
    if (failed(lhsMemref))
      return failure();

    FailureOr<Value> rhsMemref =
        getBuffer(rewriter, aiceMatMulOp.getRhs(), options);
    if (failed(rhsMemref))
      return failure();

    auto resultMemrefType =
        bufferization::getBufferType(aiceMatMulOp.getResult(), options);
    if (failed(resultMemrefType))
      return failure();

    // Create a new AiceMatMulOp with the bufferized operands.
    auto newOp = rewriter.create<kestrel::AiceMatMulOp>(
        op->getLoc(), llvm::cast<MemRefType>(*resultMemrefType), *lhsMemref,
        *rhsMemref);

    replaceOpWithBufferizedValues(rewriter, op, newOp.getResult());
    return success();
  }
};

} // namespace kestrel
} // namespace mlir

void mlir::kestrel::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, kestrel::KestrelDialect *dialect){
    DMALoadOp::attachInterface<DMALoadOpInterface>(*ctx);
    DMAStoreOp::attachInterface<DMAStoreOpInterface>(*ctx);
    AiceMatMulOp::attachInterface<AiceMatMulOpInterface>(*ctx);
  });
}