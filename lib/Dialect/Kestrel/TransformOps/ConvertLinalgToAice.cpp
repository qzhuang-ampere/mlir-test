#include "include/Dialect/Kestrel/TransformOps/Passes.h"
#include "include/Dialect/Kestrel/TransformOps/KestrelOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

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

    std::vector<Dialect *> dialects = context->getLoadedDialects();

    // Create a new AiceMatMulOp with the same attributes as the original.
    auto newOp = rewriter.create<kestrel::AiceMatMulOp>(
        op.getLoc(), op.getResultTypes(), op.getInputs()[0], op.getInputs()[1]);
    rewriter.replaceOp(op, newOp);
    return success();
  }
};

void populateLinalgToAiceConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<LinalgMatmulToAiceMatul>(patterns.getContext());
}

void ConvertLinalgToAicePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateLinalgToAiceConversionPatterns(patterns);

  ConversionTarget target(getContext());
  target.addLegalDialect<kestrel::KestrelDialect>();

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}
