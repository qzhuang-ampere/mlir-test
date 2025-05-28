//===- KestrelTransformOps.td - Transform dialect extension -----------------===//
// This file defines a dialect extension for the Transform dialect
//===----------------------------------------------------------------------===//

#ifndef KESTREL_TRANSFORM_OPS
#define KESTREL_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

///===----------------------------------------------------------------------===//
// Kestrel Transform dialect extension
// Outline the handle to a function with a unique name
//===----------------------------------------------------------------------===//
def OutlineWithUniqName : Op<Transform_Dialect, "kestrel.loop.outline_with_uniq_name",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Generates a unique function name for a outlined function call";
  let description = [{
    Generates a unique function name for a outlined function call.
    The generated name is guaranteed to be unique within the current module.
    The generated name is based on the original function name and a unique suffix.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                   StrAttr:$func_name);

  let results = (outs TransformHandleTypeInterface:$function,
                      TransformHandleTypeInterface:$call);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
}

def MergeInnerScfForAll : Op<Transform_Dialect, "kestrel.loop.merge_inner_scf_for_all",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Merges inner scf.forall operations into a single scf.forall";
  let description = [{
    Merges all inner scf.forall operations into a single scf.forall operation.
    This is useful for optimizing the loop structure and reducing the number of
    nested loops in the generated code.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let results = (outs TransformHandleTypeInterface:$merged_forall);

  let assemblyFormat = "$target attr-dict `:` functional-type(operands, results)";
}
#endif // KESTREL_TRANSFORM_OPS
