//===- UniqueFuncNameExt.td - Transform dialect extension -----------------===//
// This file defines a dialect extension for the Transform dialect
//===----------------------------------------------------------------------===//

#ifndef KESTREL_TRANSFORM_OPS
#define KESTREL_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

///===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
def OutlineWithUniqName : Op<Transform_Dialect, "kestrel.loop.outline_with_uniq_name",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
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

#endif // KESTREL_TRANSFORM_OPS
