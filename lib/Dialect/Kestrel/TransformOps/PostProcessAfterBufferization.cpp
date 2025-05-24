#include "include/Dialect/Kestrel/TransformOps/Passes.h"
#include "include/Dialect/Kestrel/TransformOps/KestrelOps.h"
#include "include/Dialect/Kestrel/TransformOps/common.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "kestrel-post-process-after-bufferization"

namespace mlir {
  #define GEN_PASS_DEF_POSTPROCESSAFTERBUFFERIZATION
  #include "Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::kestrel;

namespace {
class PostProcessAfterBufferizationPass
    : public impl::PostProcessAfterBufferizationBase<PostProcessAfterBufferizationPass> {
  using Base::Base;

  void runOnOperation() override;

private:

};
}

struct MemrefSubViewCopyToDMAStore : public OpConversionPattern<memref::CopyOp> {
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MLIRContext *context = rewriter.getContext();

    auto prevOp = op.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!prevOp) {
      return failure();
    }

    auto newOp = rewriter.create<kestrel::DMAStoreOp>(
        op.getLoc(), op.getSource(), prevOp.getSource(), prevOp.getOffsets(),
        prevOp.getStrides(), prevOp.getSizes(), prevOp.getStaticOffsets(),
        prevOp.getStaticStrides(), prevOp.getStaticSizes());

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

void populateMemrefToDMAConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<MemrefSubViewCopyToDMAStore>(patterns.getContext());
}

void PostProcessAfterBufferizationPass::runOnOperation() {
  if (loadOnly == true) {
    llvm::dbgs() << "PostProcessAfterBufferizationPass: Load into context only, not doing anything\n";
    return;
  }
  else {
    llvm::dbgs() << "PostProcessAfterBufferizationPass: executing pass\n";
  }

  ConversionTarget target(getContext());
  target.addLegalDialect<kestrel::KestrelDialect>();
  target.addIllegalOp<memref::CopyOp>();

  // The pass logic goes here.
  // For example, you can perform some transformations on the operation.
  auto module = getOperation();
  if (llvm::isa<ModuleOp>(module) == false && llvm::isa<mlir::func::FuncOp>(module) == false) {
    llvm::errs() << "Error: Cannot apply PostProcessAfterBufferization pass to an op which is not a ModuleOp for FuncOp\n";
  }

  module->walk([&](Operation *opPtr) {
    if (auto function = mlir::dyn_cast<func::FuncOp>(opPtr)) {
      if (function.getName() == kMainFunctionName) {
        RewritePatternSet patterns(&getContext());
        populateMemrefToDMAConversionPatterns(patterns);
        if (failed(applyPartialConversion(function, target, std::move(patterns)))) {
          signalPassFailure();
        }
      }

      function.walk([&](Operation *opPtr) {
      });
    }
  });

}
