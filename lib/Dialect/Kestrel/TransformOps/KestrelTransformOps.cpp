//===- UniqueFuncNameExt.td - Transform dialect extension -----------------===//
// This file defines a dialect extension for the Transform dialect
//===----------------------------------------------------------------------===//

#include "KestrelTransformOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  static const int length = 4; // Desired length of the random string

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
  llvm::dbgs() << nameWithPostfix << "\n";

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

void registerKestrelTransformOps(::mlir::DialectRegistry &registry) {
  registry.addExtensions<KestrelTransformOps>();
}
