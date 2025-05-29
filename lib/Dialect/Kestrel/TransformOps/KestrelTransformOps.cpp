//===- KestrealTransformOps.cpp - Transform dialect extension -----------------===//
// This file defines a dialect extension for the Transform dialect
//===----------------------------------------------------------------------===//

#include "KestrelOps.h"
#include "KestrelTransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <random>
#include <ctime>

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

using namespace mlir;

namespace {
std::string generateRandomString() {
  // Seed the random number generator
  static const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  static std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  static std::uniform_int_distribution<size_t> distribution(0, characters.size() - 1);
  static const int length = 8; // Desired length of the random string

  std::string randomString = "_";

  for (int i = 0; i < length; ++i) {
    randomString += characters[distribution(generator)];
  }

  return randomString;
}

static scf::ExecuteRegionOp wrapInExecuteRegion(RewriterBase &b,
                                                Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      b.create<scf::ExecuteRegionOp>(op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = b.cloneWithoutRegions(*op);
    Region &clonedRegion = clonedOp->getRegions().front();
    assert(clonedRegion.empty() && "expected empty region");
    b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                         clonedRegion.end());
    b.create<scf::YieldOp>(op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}
}

::mlir::DiagnosedSilenceableFailure mlir::transform::OutlineWithUniqName::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {

  auto target = getTarget();
  llvm::StringRef funcName = getFuncName();
  std::string nameWithPostfix = funcName.str() + generateRandomString();

  SmallVector<Operation *> functions;
  SmallVector<Operation *> calls;
  DenseMap<Operation *, SymbolTable> symbolTables;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    Location location = target->getLoc();
    Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(target);
    scf::ExecuteRegionOp exec = wrapInExecuteRegion(rewriter, target);
    if (!exec) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "failed to outline";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    func::CallOp call;
    FailureOr<func::FuncOp> outlined =outlineSingleBlockRegion(
        rewriter, location, exec.getRegion(), nameWithPostfix, &call);

    if (failed(outlined))
      return emitDefaultDefiniteFailure(target);

    if (symbolTableOp) {
      SymbolTable &symbolTable =
          symbolTables.try_emplace(symbolTableOp, symbolTableOp)
              .first->getSecond();
      symbolTable.insert(*outlined);
      call.setCalleeAttr(FlatSymbolRefAttr::get(*outlined));
    }
    functions.push_back(*outlined);
    calls.push_back(call);
  }
  results.set(cast<OpResult>(getFunction()), functions);
  results.set(cast<OpResult>(getCall()), calls);


  return DiagnosedSilenceableFailure::success();
}

mlir::Value getSingleAttributeValueFromOpFoldResult(std::optional<mlir::OpFoldResult> &ofr) {
  llvm_unreachable("Not implemented");
  return nullptr;
}

scf::ForallOp mergeInnerScfForAll(
    ::mlir::RewriterBase &rewriter, scf::ForallOp forallOp, scf::ForallOp innerForallOp) {
  if (forallOp.getNumOperands() != 1)
    return nullptr;

  // Get the loop variable from innerForallOp
  auto lowerbound = innerForallOp.getSingleLowerBound();
  auto upperbound = innerForallOp.getSingleUpperBound();

  auto attrLb = cast<IntegerAttr>(cast<Attribute>(*lowerbound));
  if (!attrLb) {
    llvm::dbgs() << innerForallOp << "expected lower bound to be an IntegerAttr\n";
    return nullptr;
  }
  else {
    // If the lower bound is not zero, return nullptr as we only support zero lower bounds
    if (attrLb.getInt() != 0) {
      llvm::dbgs() << innerForallOp << "expected lower bound to be zero\n";
      return nullptr;
    }
  }

  // check upper bound must be attr
  auto attrUb = cast<IntegerAttr>(cast<Attribute>(*upperbound));
  int64_t innerUbVal = 0;
  if (!attrUb) {
    llvm::dbgs() << innerForallOp << "expected upper bound to be an IntegerAttr\n";
    return nullptr;
  }
  else {
    // If the upper bound is not one, return nullptr as we only support one upper bounds
    if (attrUb.getInt() <= 0) {
      llvm::dbgs() <<  innerForallOp << "expected upper bound to be greater than zero\n";
      return nullptr;
    }
    else {
      innerUbVal = attrUb.getInt();
    }
  }

  // get outter loops upper bound value
  auto outerUpperBound = forallOp.getMixedUpperBound()[1];
  auto attrOuterUb = cast<IntegerAttr>(cast<Attribute>(outerUpperBound));
  int64_t outerUbVal = 0;
  if (!attrOuterUb) {
    llvm::dbgs() << forallOp << "expected outer upper bound to be an IntegerAttr\n";
    return nullptr;
  }
  else {
      outerUbVal = attrOuterUb.getInt();
  }

  int64_t newUpperBound = outerUbVal * innerUbVal;
  auto newUpperBounds = forallOp.getMixedUpperBound();
  newUpperBounds[1] = rewriter.getIndexAttr(newUpperBound);
  // get the location before forallOp

  // check if the innerForallOp has a single output
  // Create a new scf::ForallOp with updated upper bounds

  rewriter.setInsertionPoint(forallOp);
  auto newForallOp = rewriter.create<scf::ForallOp>(
      forallOp.getLoc(), forallOp.getMixedLowerBound(),
      newUpperBounds, forallOp.getMixedStep(), forallOp.getOutputs(),
      forallOp.getMapping(),
      /*bodyBuilderFn=*/[](OpBuilder &, Location, ValueRange) {});

  // get induction variable from the outer newForallOp
  auto outerInductionVar0 = newForallOp.getInductionVars()[0];
  auto outerInductionVar1 = newForallOp.getInductionVars()[1];

  rewriter.setInsertionPointToStart(newForallOp.getBody());
  auto intVal = rewriter.create<arith::ConstantIndexOp>(
      forallOp.getLoc(), innerUbVal);

  auto divOp = rewriter.create<arith::DivUIOp>(
      forallOp.getLoc(), outerInductionVar1, intVal);

  auto moduloOp = rewriter.create<arith::RemUIOp>(
      forallOp.getLoc(), outerInductionVar1, intVal);

  auto reduction_input = innerForallOp.getResults()[0];

  // get the shape of reduction_input, discard the last dimension, create a new tensor with the new shape
  auto reductionType = cast<RankedTensorType>(reduction_input.getType());
  auto reductionShape = reductionType.getShape();
  SmallVector<int64_t> newReductionShape(reductionShape.begin(),
                                         reductionShape.end() - 1);
  auto newReductionType = RankedTensorType::get(
      newReductionShape, reductionType.getElementType(),
      reductionType.getEncoding());
  // create a new tensor with the new shape
  Value newReductionInput = rewriter.create<tensor::EmptyOp>(
      forallOp.getLoc(), newReductionShape, newReductionType.getElementType(),
      newReductionType.getEncoding());
  // create linalg.fill op to fill the new tensor with constant 0
  Value initValue = rewriter.create<arith::ConstantOp>(
      forallOp.getLoc(), rewriter.getZeroAttr(newReductionType.getElementType()));
  auto newLinalgFill = rewriter.create<linalg::FillOp>(forallOp.getLoc(),
                                                       initValue,
                                                       newReductionInput);

  // insert a terminator to the new ForallOp body
  auto yieldOp = rewriter.create<scf::YieldOp>(forallOp.getLoc(), newForallOp.getOutputs());

  // Copy all operations from the old outer loop body to the new ForallOp body
  Block *oldBody = forallOp.getBody();
  Block *newBody = newForallOp.getBody();
  auto innerOutType = innerForallOp.getRegionOutArgs()[0];
  auto outerOut = forallOp.getRegionOutArgs()[0];
  auto newOut = newForallOp.getRegionOutArgs()[0];

  rewriter.setInsertionPoint(yieldOp);

  auto oldOutInductionVar = forallOp.getInductionVars()[1];
  auto oldInnerInductionVar = innerForallOp.getInductionVars()[0];
  oldOutInductionVar.replaceAllUsesWith(divOp.getResult());
  oldInnerInductionVar.replaceAllUsesWith(moduloOp.getResult());
  outerOut.replaceAllUsesWith(newOut);

  // get the innerForallOp's body, then get the first op in the body
  auto firstOp = *(innerForallOp.getBody()->getOps<tensor::ExtractSliceOp>().begin());
  firstOp.getResult().replaceAllUsesWith(newLinalgFill.getResult(0));

  Value outputForReduce = nullptr;
  Value gemmOutput = nullptr;
  for (auto &op : llvm::make_early_inc_range(*oldBody)) {
    // Do not move the terminator (scf.yield), as the new op already has one
    if (llvm::isa<scf::YieldOp>(op))
      continue;

    else if (llvm::isa<scf::ForallOp>(op)) {
      scf::ForallOp innerOp = llvm::cast<scf::ForallOp>(op);
      Block *innerBody = innerOp.getBody();

      for (auto &op1 : llvm::make_early_inc_range(*innerBody)) {
        if (&op1 == firstOp.getOperation()) {
          // first op is not going to be copied.
          continue;
        }
        else if (llvm::isa<scf::InParallelOp>(op1)) {
          // skip the forall.in_parallel op with insert_slice
          continue;
        }
        else if (llvm::isa<scf::YieldOp>(op1)) {
          continue;
        }
        else {
          if (llvm::isa<linalg::MatmulOp>(op1)) {
            gemmOutput = op1.getResult(0);
          }
          rewriter.moveOpBefore(&op1, yieldOp.getOperation());
        }
      }
      continue;
    }
    else if (llvm::isa<tensor::EmptyOp>(op)) {
      continue;
    }
    else if (llvm::isa<linalg::FillOp>(op)) {
      continue;
    }
    else if (llvm::isa<linalg::ReduceOp>(op)) {
      // create a dma reduce op on it
      auto reduceOp = llvm::cast<linalg::ReduceOp>(op);
      auto src = reduceOp.getInputs()[0];
      auto dst = reduceOp.getInits()[0];
      llvm::SmallVector<OpFoldResult> offsets;
      offsets.push_back(rewriter.getIndexAttr(0));
      offsets.push_back(rewriter.getIndexAttr(0));
      llvm::SmallVector<mlir::Value> dynamicOffsets;
      llvm::SmallVector<int64_t> staticOffsets;
      dispatchIndexOpFoldResults(offsets, dynamicOffsets,
                                 staticOffsets);

      llvm::SmallVector<OpFoldResult> strides;
      strides.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
      llvm::SmallVector<mlir::Value> dynamicStrides;
      llvm::SmallVector<int64_t> staticStrides;
      dispatchIndexOpFoldResults(strides, dynamicStrides,
                                 staticStrides);

      llvm::SmallVector<OpFoldResult> sizes;
      sizes.push_back(rewriter.getIndexAttr(reductionShape[0]));
      sizes.push_back(rewriter.getIndexAttr(reductionShape[1]));
      llvm::SmallVector<mlir::Value> dynamicSizes;
      llvm::SmallVector<int64_t> staticSizes;
      dispatchIndexOpFoldResults(sizes, dynamicSizes,
                                 staticSizes);
      auto result = rewriter.create<kestrel::DMAReduceOp>(forallOp.getLoc(),
                                                          gemmOutput,
                                                          dst,
                                                          dynamicOffsets,
                                                      dynamicStrides,
                                                      dynamicSizes,
                                                      staticOffsets,
                                                      staticStrides,
                                                      staticSizes);

      reduceOp.getResult(0).replaceAllUsesWith(outputForReduce);
      continue;
    }
    else {
      if (llvm::isa<tensor::ExtractSliceOp>(op) && op.getOperand(0) == newOut) {
        outputForReduce = op.getResult(0);
      }
      rewriter.moveOpBefore(&op, yieldOp.getOperation());
    }
  }
  // rewriter.inlineRegionBefore(forallOp.getRegion(), newForallOp.getRegion(),
  //                             newForallOp.getRegion().end());
  // forallOp.getResult(0).replaceAllUsesWith(newForallOp.getResult(0));
  // for (auto u : forallOp->getUsers()) {
  //   u->dump();
  // }
  // Todo: remove all yieldOps with block->end()
  yieldOp.erase();
  rewriter.replaceOp(forallOp, newForallOp);
  return newForallOp;
}

::mlir::DiagnosedSilenceableFailure mlir::transform::MergeInnerScfForAll::apply(
    ::mlir::transform::TransformRewriter &rewriter,
    ::mlir::transform::TransformResults &results,
    ::mlir::transform::TransformState &state) {

  auto target = getTarget();
  SmallVector<Operation *> mergedOps;
  for (Operation *targetOp : state.getPayloadOps(getTarget())) {
    Location location = targetOp->getLoc();
    scf::ForallOp forallOp = dyn_cast<scf::ForallOp>(targetOp);
    if (!forallOp) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "expected scf.forall op";
      diag.attachNote(targetOp->getLoc()) << "target op";
      return diag;
    }
    scf::ForallOp innerForallOp = nullptr;
    forallOp.getBody()->walk([&](scf::ForallOp op) {
      if (op != forallOp && !innerForallOp) {
        innerForallOp = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!innerForallOp) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "no inner scf.forall op found";
      diag.attachNote(location) << "target op";
      return diag;
    }

    scf::ForallOp merged = mergeInnerScfForAll(rewriter, forallOp, innerForallOp);
    if (!merged) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "failed to merge inner scf.forall";
      diag.attachNote(location) << "target op";
      return diag;
    }
    mergedOps.push_back(merged);
  }
  results.set(cast<OpResult>(getMergedForall()), mergedOps);
  return DiagnosedSilenceableFailure::success();
}

void registerKestrelTransformOps(::mlir::DialectRegistry &registry) {
  registry.addExtensions<KestrelTransformOps>();
}
