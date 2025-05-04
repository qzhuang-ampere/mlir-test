add_llvm_executable(kestrel-opt kestrel-opt.cpp)

target_link_libraries(kestrel-opt PRIVATE
  KestrelIR
  MLIRIR
  MLIRMlirOptMain
  MLIRSideEffectInterfaces)