set(LLVM_TARGET_DEFINITIONS KestrelTransformOps.td)
mlir_tablegen(KestrelTransformOps.h.inc -gen-op-decls)
mlir_tablegen(KestrelTransformOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(KestrelTransformOpsIncGen)

add_mlir_doc(KestrelTransformOpsDoc KestrelTransformOpsDoc Kestrel/ -gen-op-doc)