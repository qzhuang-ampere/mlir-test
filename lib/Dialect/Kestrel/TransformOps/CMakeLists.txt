add_mlir_library(
  KestrelIR

  KestrelOps.cpp
  KestrelTransformOps.cpp
  ConvertLinalgToAice.cpp
  PostProcessAfterBufferization.cpp
  BufferizableOpInterfaceImpl.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Kestrel/TransformOps

  DEPENDS
  KestrelOpsIncGen
  KestrelTransformOpsIncGen
  KestrelPassIncGen

  LINK_LIBS PRIVATE
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
)

set(MLIR_KESTREL_LIBS
  KestrelIR
)