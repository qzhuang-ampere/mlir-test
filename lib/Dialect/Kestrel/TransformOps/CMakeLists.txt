add_mlir_library(
  KestrelTransformOps

  KestrelTransformOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/Kestrel/TransformOps

  DEPENDS
  KestrelTransformOpsIncGen

  LINK_LIBS PRIVATE
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
)
