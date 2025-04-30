//===- UniqueFuncNameExt.td - Transform dialect extension -----------------===//
// This file defines a dialect extension for the Transform dialect
//===----------------------------------------------------------------------===//

#include "KestrelTransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

class KestrelTransformOps
    : public ::mlir::transform::TransformDialectExtension<KestrelTransformOps> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(KestrelTransformOps)

  using Base::Base;

  void init();
};

void KestrelTransformOps::init() {
  declareGeneratedDialect<::mlir::scf::SCFDialect>();
  declareGeneratedDialect<::mlir::func::FuncDialect>();

  registerTransformOps<
#define GET_OP_LIST
#include "KestrelTransformOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "KestrelTransformOps.cpp.inc"

static void updateCallee(mlir::func::CallOp call, llvm::StringRef newTarget) {
  call.setCallee(newTarget);
}

::mlir::DiagnosedSilenceableFailure mlir::transform::OutlineWithUniqName::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {


  auto target = getTarget();

  return DiagnosedSilenceableFailure::success();
}

void mlir::transform::OutlineWithUniqName::getEffects(
    ::llvm::SmallVectorImpl<::mlir::MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);

  modifiesPayload(effects);
}

void registerKestrelTransformOps(::mlir::DialectRegistry &registry) {
  registry.addExtensions<KestrelTransformOps>();
}
