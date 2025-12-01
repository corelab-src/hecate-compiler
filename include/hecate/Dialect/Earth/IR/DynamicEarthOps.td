#ifndef HECATE_DYNAMIC_ARITH_OPS
#define HECATE_DYNAMIC_ARITH_OPS

include "hecate/Dialect/Earth/IR/EarthOps.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "hecate/Dialect/Earth/IR/HEParameterInterface.td"
include "hecate/Dialect/Earth/IR/ForwardManagementInterface.td"

def LevelCastOp : Op<Earth_Dialect, "level_cast", [DeclareOpInterfaceMethods<OpAsmOpInterface>,DeclareOpInterfaceMethods<CastOpInterface>]> {
  let arguments = (ins AnyTensor:$source);
  let results = (outs AnyTensor:$dest);
  let summary = "Cast the type (scale, level, encrypted) of the tensor to the target type";
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
  let hasCanonicalizer = 1;
}

def ScaleCastOp: Unary_Earth_Op<"scale_cast">  {
  let arguments = (ins HETensor:$value, I64Attr:$integer);
  let results = (outs HETensor);
  let hasCanonicalizer = 1;
}
#endif
