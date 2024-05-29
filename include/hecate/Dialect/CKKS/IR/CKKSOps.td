
#ifndef HECATE_CKKS_OPS
#define HECATE_CKKS_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "hecate/Dialect/CKKS/IR/PolyTypeInterface.td"
/* include "hecate/Dialect/Earth/IR/HEParameterInterface.td" */
/* include "hecate/Dialect/Earth/IR/ForwardManagementInterface.td" */


def CKKS_Dialect : Dialect {
  let name = "ckks";
  let cppNamespace = "::hecate::ckks";
  /* let useFoldAPI= kEmitFoldAdaptorFolder; */
  let useDefaultTypePrinterParser = 1;
}

def PolyType : TypeDef<CKKS_Dialect, "Poly", [DeclareTypeInterfaceMethods<PolyTypeInterface>, DeclareTypeInterfaceMethods<MemRefElementTypeInterface>]> {
  let summary = "CKKS Polynomial Type with # of polynomial and level";
  let description = [{
    A type for CKKS API level optimization.
  }];
  let mnemonic = "poly";
  let parameters = (ins "unsigned":$num_poly, "unsigned":$level);
  let assemblyFormat = "`<` $num_poly `*` $level `>`";
  let genVerifyDecl = 0;
}

def PolyTensor : TensorOf<[PolyType]>;

class CKKS_Op<string mnemonic, list<Trait> traits = []> :
Op<CKKS_Dialect, mnemonic, !listconcat(traits, [DeclareOpInterfaceMethods<DestinationStyleOpInterface>, DeclareOpInterfaceMethods<InferTypeOpInterface>, DeclareOpInterfaceMethods<HEVMOpInterface>])> {
  let results = (outs PolyTensor);
  string arithClassDefinition = [{
    // - ScaleOpInterface Non-default methods
    // - isSingle, isConsume is remaining methods
    /* std::pair<int64_t, int64_t> $cppClass::getDpsInitsPositionRange() { return {0, 1};} */
    ::mlir::MutableOperandRange $cppClass::getDpsInitsMutable() { return {getOperation(), 0, 1};}
  }];
  let extraClassDefinition = arithClassDefinition;
}

/* ######### Zero-operand Operations ########## */

def EncodeOp : CKKS_Op<"encode",[]> {

  let arguments = (ins PolyTensor:$dst, I64Attr:$value, I64Attr:$scale, I64Attr:$level);
  // AnyAttr because of constant hoisting
  let results = (outs PolyTensor);


  code extraClassDeclaration = [{
    static bool isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs) ;
    int64_t getNumOperands();
  }];

  code extraClassDefinition = arithClassDefinition # [{
    bool $cppClass::isCompatibleReturnTypes(::mlir::TypeRange lhs, ::mlir::TypeRange rhs){
      return rhs.back().dyn_cast<PolyTypeInterface>().getNumPoly() == 1;
    }
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 0;
      op.dst = plainMap[getDst()];
      op.lhs = getValue();
      /* op.rhs = (getLevel() << 8) + getScale(); */
      op.rhs = (getLevel() << 10) + getScale();
      /* op.rhs = getLevel() << 10 ; */
      return op;
    }
    int64_t $cppClass::getNumOperands() {
      return 1;
    }
  }];
}

/* ######### Unary Operations ########## */


def RotateCOp : CKKS_Op<"rotatec", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src, DenseI64ArrayAttr:$offset);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 1;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getSrc()];
      op.rhs = getOffset()[0];
      return op;
    }
  }];
}
def NegateCOp : CKKS_Op<"negatec", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 2;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getSrc()];
      op.rhs = 0;
      return op;
    }
  }];
}

/* ######## Scale Management Operations ####### */

def RescaleCOp : CKKS_Op<"rescalec", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 3;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getSrc()];
      op.rhs = 0;
      return op;
    }
  }];
}
def ModswitchCOp : CKKS_Op<"modswitchc", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src, I64Attr:$downFactor);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 4;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getSrc()];
      op.rhs = getDownFactor();
      return op;
    }
  }];
}
def UpscaleCOp : CKKS_Op<"upscalec", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src, I64Attr:$upFactor);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 5;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getSrc()];
      op.rhs = getUpFactor();
      return op;
    }
  }];
}

def BootstrapCOp : CKKS_Op<"bootstrapc", [SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$src, I64Attr:$level);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 10;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getSrc()];
      op.rhs = getLevel();
      return op;
    }
  }];
}

/* ######### Binary Operations ########## */

def AddCCOp: CKKS_Op<"addcc", [Commutative, SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 6;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getLhs()];
      op.rhs = cipherMap[getRhs()];
      return op;
    }
  }];
}
def AddCPOp: CKKS_Op<"addcp", [Commutative, SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 7;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getLhs()];
      op.rhs = plainMap[getRhs()];
      return op;
    }
  }];
}
def MulCCOp: CKKS_Op<"mulcc", [Commutative, SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 8;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getLhs()];
      op.rhs = cipherMap[getRhs()];
      return op;
    }
  }];
}
def MulCPOp: CKKS_Op<"mulcp", [Commutative, SameOperandsAndResultShape]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs, PolyTensor:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 9;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getLhs()];
      op.rhs = plainMap[getRhs()];
      return op;
    }
  }];
}
/* def RotateCOp : CKKS_Op<"rotatec", [Commutative, SameOperandsAndResultShape]> { */
/*   let arguments = (ins PolyTensor:$dst, PolyTensor:$src, Index:$offset); */
/*   code extraClassDefinition = arithClassDefinition # [{ */
/*     HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap, llvm::DenseMap<mlir::Value, int64_t> cipherMap){ */
/*       HEVMOperation op; */
/*       OpBuilder builder(getOperation()); */
/*       op.opcode = 1; */
/*       op.dst = cipherMap[getDst()]; */
/*       op.lhs = cipherMap[getSrc()]; */
/*       /1* auto intOffset = getOffset().getResult().getType().dyn_cast<IntegerType>(); *1/ */
/*       op.rhs = getOperand(2).getType().dyn_cast<IntegerType>(); */
/*       /1* op.rhs = 1; *1/ */
/*       return op; */
/*     } */
/*   }]; */
/* } */
/* ######### Memory Management Operations ########## */
def CopyCOp: CKKS_Op<"copyc", [SameOperandsAndResultType]> {
  let arguments = (ins PolyTensor:$dst, PolyTensor:$lhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 200;
      op.dst = cipherMap[getDst()];
      op.lhs = cipherMap[getLhs()];
      op.rhs = 0;
      return op;
    }
  }];
}


/* ######### Index Arith Operations ########## */
def EmptyIOp: CKKS_Op<"emtpyi", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$dst);
  let results = (outs Index:$result);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 99;
      op.dst = 0;
      op.lhs = 0;
      op.rhs = 0;
      return op;
    }
  }];
}

def ConstIOp: CKKS_Op<"constanti", [SameOperandsAndResultType]> {
  let arguments = (ins Index:$dst, I64Attr:$value);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 100;
      op.dst = integerMap[getDst()];
      op.lhs = getValue();
      op.rhs = 0;
      return op;
    }
  }];
}


/* ######### Index Arith Operations ########## */

def AddIOp: CKKS_Op<"addi", [Commutative, SameOperandsAndResultType]> {
  let arguments = (ins Index:$dst, Index:$lhs, Index:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 101;
      op.dst = integerMap[getDst()];
      op.lhs = integerMap[getLhs()];
      op.rhs = integerMap[getRhs()];
      return op;
    }
  }];
}

def SubIOp: CKKS_Op<"subi", [Commutative, SameOperandsAndResultType]> {
  let arguments = (ins Index:$dst, Index:$lhs, Index:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 102;
      op.dst = integerMap[getDst()];
      op.lhs = integerMap[getLhs()];
      op.rhs = integerMap[getRhs()];
      return op;
    }
  }];
}
def RemSIOp: CKKS_Op<"remsi", [Commutative, SameOperandsAndResultType]> {
  let arguments = (ins Index:$dst, Index:$lhs, Index:$rhs);
  code extraClassDefinition = arithClassDefinition # [{
    HEVMOperation $cppClass::getHEVMOperation (llvm::DenseMap<mlir::Value, int64_t> plainMap,
        llvm::DenseMap<mlir::Value, int64_t> cipherMap,
        llvm::DenseMap<mlir::Value, int64_t> integerMap){
      HEVMOperation op;
      op.opcode = 103;
      op.dst = integerMap[getDst()];
      op.lhs = integerMap[getLhs()];
      op.rhs = integerMap[getRhs()];
      return op;
    }
  }];
}
#endif
