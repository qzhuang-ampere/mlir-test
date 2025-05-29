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
    return true;
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

struct DMAStoreWithResultOpInterface
    : public BufferizableOpInterface::ExternalModel<DMAStoreWithResultOpInterface,
                                                    kestrel::DMAStoreWithResultOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return true;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dmaStoreOp = cast<kestrel::DMAStoreWithResultOp>(op);
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

    auto newOp = rewriter.create<kestrel::DMAStoreWithResultOp>(
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

    FailureOr<Value> outMemref =
        getBuffer(rewriter, aiceMatMulOp.getOut(), options);
    if (failed(outMemref))
      return failure();

    auto resultMemrefType =
        bufferization::getBufferType(aiceMatMulOp.getResult(), options);
    if (failed(resultMemrefType))
      return failure();

    // Create a new AiceMatMulOp with the bufferized operands.
    auto newOp = rewriter.create<kestrel::AiceMatMulOp>(
        op->getLoc(), llvm::cast<MemRefType>(*resultMemrefType), *lhsMemref,
        *rhsMemref, *outMemref);

    replaceOpWithBufferizedValues(rewriter, op, newOp.getResult());
    return success();
  }
};

struct DMAReduceOpInterface
    : public BufferizableOpInterface::ExternalModel<DMAReduceOpInterface,
                                                    kestrel::DMAReduceOp> {
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
    return {};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto dmaReduceOp = cast<kestrel::DMAReduceOp>(op);
    FailureOr<Value> srcMemref =
        getBuffer(rewriter, dmaReduceOp.getSource(), options);
    if (failed(srcMemref))
      return failure();

    FailureOr<Value> dstMemref =
        getBuffer(rewriter, dmaReduceOp.getDest(), options);
    if (failed(dstMemref))
      return failure();

    auto newOp = rewriter.create<kestrel::DMAReduceOp>(
        dmaReduceOp->getLoc(), *srcMemref, *dstMemref,
        dmaReduceOp.getOffsets(),
        dmaReduceOp.getSizes(),
        dmaReduceOp.getStrides(),
        dmaReduceOp.getStaticOffsets(),
        dmaReduceOp.getStaticSizes(),
        dmaReduceOp.getStaticStrides());

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace kestrel
} // namespace mlir

void mlir::kestrel::registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, kestrel::KestrelDialect *dialect){
    DMALoadOp::attachInterface<DMALoadOpInterface>(*ctx);
    DMAStoreWithResultOp::attachInterface<DMAStoreWithResultOpInterface>(*ctx);
    AiceMatMulOp::attachInterface<AiceMatMulOpInterface>(*ctx);
    DMAReduceOp::attachInterface<DMAReduceOpInterface>(*ctx);
  });
}