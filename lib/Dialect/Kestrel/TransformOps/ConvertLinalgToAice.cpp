#include "include/Dialect/Kestrel/TransformOps/Passes.h"
#include "include/Dialect/Kestrel/TransformOps/KestrelOps.h"
#include "include/Dialect/Kestrel/TransformOps/common.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "kestrel-pass"

namespace mlir {
  #define GEN_PASS_DEF_CONVERTLINALGTOAICE
  #include "Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace {
class ConvertLinalgToAicePass
    : public impl::ConvertLinalgToAiceBase<ConvertLinalgToAicePass> {
  void runOnOperation() override;
};
}

struct LinalgMatmulToAiceMatul : public OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MLIRContext *context = rewriter.getContext();

    // Create a new AiceMatMulOp with the same attributes as the original.
    auto newOp = rewriter.create<kestrel::AiceMatMulOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs()[0], op.getInputs()[1]);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TensorExtractSliceToDMALoad : public OpConversionPattern<tensor::ExtractSliceOp> {
  using OpConversionPattern<tensor::ExtractSliceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Create a new DMALoadOp with the same attributes as the original.
    auto mixedSizes = op.getMixedSizes();
    auto mixedStrides = op.getMixedStrides();
    auto mixedOffsets = op.getMixedOffsets();

    auto newOp = rewriter.create<kestrel::DMALoadOp>(
        op.getLoc(), op.getResultType(), op.getSource(), op.getOffsets(),
        op.getStrides(), op.getSizes(), op.getStaticOffsets(),
        op.getStaticStrides(), op.getStaticSizes());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct TensorInsertSliceToDMAStore : public OpConversionPattern<tensor::InsertSliceOp> {
  using OpConversionPattern<tensor::InsertSliceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Create a new DMAStoreOp with the same attributes as the original.
    auto mixedSizes = op.getMixedSizes();
    auto mixedStrides = op.getMixedStrides();
    auto mixedOffsets = op.getMixedOffsets();

    auto newOp = rewriter.create<kestrel::DMAStoreOp>(
        op.getLoc(), op.getResultType(), op.getSource(), op.getDest(), op.getOffsets(),
        op.getStrides(), op.getSizes(), op.getStaticOffsets(),
        op.getStaticStrides(), op.getStaticSizes());
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

void populateLinalgToAiceConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgMatmulToAiceMatul>(patterns.getContext());
}

void populateGlobalTensorMoveToDMAConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TensorExtractSliceToDMALoad>(patterns.getContext());
  patterns.add<TensorInsertSliceToDMAStore>(patterns.getContext());
}

void ConvertLinalgToAicePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLinalgToAiceConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<kestrel::KestrelDialect>();

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  auto module = getOperation();
  if (llvm::isa<ModuleOp>(module) == false) {
    llvm::errs() << "Error: Cannot apply ConvertLinalgToAice pass to an op which is not a ModuleOp\n";
  }

  // Loop all FuncOps in the module and print their names.
  module->walk([&](Operation *opPtr) {
    if (auto function = mlir::dyn_cast<func::FuncOp>(opPtr)) {
      if (function.getName().starts_with(kestrel::kConnorFunctionPrefix) ||
          function.getName().starts_with(kestrel::kAiceFunctionPrefix)) {
        LLVM_DEBUG(llvm::outs() << "Skipping function name: " << function.getName() << "\n");
      }
      else {
        LLVM_DEBUG(llvm::outs() << "Working on unction name: " << function.getName() << "\n");
        RewritePatternSet patterns(&getContext());
        populateGlobalTensorMoveToDMAConversionPatterns(patterns);
        if (failed(applyPartialConversion(function, target, std::move(patterns)))) {
          signalPassFailure();
        }
      }

      // loop all operations in the function
      function.walk([&](Operation *opPtr) {
        if (auto op = mlir::dyn_cast<tensor::ExtractSliceOp>(opPtr)) {
        }
        else {
        }
      });
    }
  });
}
