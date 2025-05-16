#include "include/Dialect/Kestrel/TransformOps/KestrelOps.h"
#include "include/Dialect/Kestrel/TransformOps/KestrelTransformOps.h"
#include "include/Dialect/Kestrel/TransformOps/Passes.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include <cstdlib>

int main(int argc, char **argv) {
  // Register all "core" dialects and our transform dialect extension.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerPassManagerCLOptions();

  // Register the transform ops
  registerKestrelTransformOps(registry);

  // Register transform interpreter pass. for the cmd line option
  mlir::transform::registerInterpreterPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerSymbolDCEPass();
  mlir::registerConvertLinalgToAicePass();
  mlir::bufferization::registerOneShotBufferizePass();
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return ::mlir::createLinalgBlockPackMatmul();
  });

  // Delegate to the MLIR utility for parsing and pass management.
  return mlir::MlirOptMain(argc, argv, "kestrel-opt", registry)
                 .succeeded()
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
