
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/IR/PolyTypeInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
//===----------------------------------------------------------------------===//
// ConstantOpInterface
//===----------------------------------------------------------------------===//
struct ConstantOpHEVMInterface
    : public hecate::ckks::HEVMOpInterface::ExternalModel<
          ConstantOpHEVMInterface, mlir::arith::ConstantOp> {
  HEVMOperation
  getHEVMOperation(Operation *op, llvm::DenseMap<mlir::Value, int64_t> plainMap,
                   llvm::DenseMap<mlir::Value, int64_t> cipherMap,
                   llvm::DenseMap<mlir::Value, int64_t> integerMap) const {
    HEVMOperation heop;
    heop.opcode = 100;
    heop.dst = integerMap[op->getResult(0)];
    heop.lhs = op->getAttrOfType<mlir::IntegerAttr>("value").getInt();

    return heop;
  }
};
struct ConstantOpDpsInterface
    : public mlir::DestinationStyleOpInterface::ExternalModel<
          ConstantOpDpsInterface, mlir::arith::ConstantOp> {

  ::mlir::MutableOperandRange getDpsInitsMutable(Operation *op) const {
    return {op, 0, 1};
  }
};

//===----------------------------------------------------------------------===//
// AddIOpInterface
//===----------------------------------------------------------------------===//

struct AddIOpHEVMInterface
    : public hecate::ckks::HEVMOpInterface::ExternalModel<AddIOpHEVMInterface,
                                                          mlir::arith::AddIOp> {
  HEVMOperation
  getHEVMOperation(Operation *op, llvm::DenseMap<mlir::Value, int64_t> plainMap,
                   llvm::DenseMap<mlir::Value, int64_t> cipherMap,
                   llvm::DenseMap<mlir::Value, int64_t> integerMap) const {
    HEVMOperation heop;
    heop.opcode = 101;
    heop.dst = integerMap[op->getResult(0)];
    heop.lhs = integerMap[op->getOperand(0)];
    heop.rhs = integerMap[op->getOperand(1)];
    return heop;
  }
};

struct AddIOpDpsInterface
    : public mlir::DestinationStyleOpInterface::ExternalModel<
          AddIOpDpsInterface, mlir::arith::AddIOp> {

  ::mlir::MutableOperandRange getDpsInitsMutable(Operation *op) const {
    return {op, 0, 1};
  }
};

//===----------------------------------------------------------------------===//
// SubOpInterface
//===----------------------------------------------------------------------===//
struct SubIOpHEVMInterface
    : public hecate::ckks::HEVMOpInterface::ExternalModel<SubIOpHEVMInterface,
                                                          mlir::arith::SubIOp> {
  HEVMOperation
  getHEVMOperation(Operation *op, llvm::DenseMap<mlir::Value, int64_t> plainMap,
                   llvm::DenseMap<mlir::Value, int64_t> cipherMap,
                   llvm::DenseMap<mlir::Value, int64_t> integerMap) const {
    HEVMOperation heop;
    heop.opcode = 102;
    heop.dst = integerMap[op->getResult(0)];
    heop.lhs = integerMap[op->getOperand(0)];
    heop.rhs = integerMap[op->getOperand(1)];
    return heop;
  }
};

struct SubIOpDpsInterface
    : public mlir::DestinationStyleOpInterface::ExternalModel<
          SubIOpDpsInterface, mlir::arith::SubIOp> {

  ::mlir::MutableOperandRange getDpsInitsMutable(Operation *op) const {
    return {op, 0, 1};
  }
  static ::mlir::LogicalResult verify(Operation *op) {
    return detail::verifyDestinationStyleOpInterface(op);
  }
};

//===----------------------------------------------------------------------===//
// RemSIOpInterface
//===----------------------------------------------------------------------===//
struct RemSIOpHEVMInterface
    : public hecate::ckks::HEVMOpInterface::ExternalModel<
          RemSIOpHEVMInterface, mlir::arith::RemSIOp> {
  HEVMOperation
  getHEVMOperation(Operation *op, llvm::DenseMap<mlir::Value, int64_t> plainMap,
                   llvm::DenseMap<mlir::Value, int64_t> cipherMap,
                   llvm::DenseMap<mlir::Value, int64_t> integerMap) const {
    HEVMOperation heop;
    heop.opcode = 103;
    heop.dst = integerMap[op->getResult(0)];
    heop.lhs = integerMap[op->getOperand(0)];
    heop.rhs = integerMap[op->getOperand(1)];
    return heop;
  }
};
struct RemSIOpDpsInterface
    : public mlir::DestinationStyleOpInterface::ExternalModel<
          RemSIOpDpsInterface, mlir::arith::RemSIOp> {

  ::mlir::MutableOperandRange getDpsInitsMutable(Operation *op) const {
    return {op, 0, 1};
  }
};

void hecate::ckks::registerArithOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.insert<arith::ArithDialect>();
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    arith::ConstantOp::attachInterface<ConstantOpHEVMInterface>(*ctx);
    /* arith::ConstantOp::attachInterface<ConstantOpDpsInterface>(*ctx); */
    arith::AddIOp::attachInterface<AddIOpHEVMInterface>(*ctx);
    /* arith::AddIOp::attachInterface<AddIOpDpsInterface>(*ctx); */
    arith::SubIOp::attachInterface<SubIOpHEVMInterface>(*ctx);
    /* arith::SubIOp::attachInterface<SubIOpDpsInterface>(*ctx); */
    arith::RemSIOp::attachInterface<RemSIOpHEVMInterface>(*ctx);
    /* arith::RemSIOp::attachInterface<RemSIOpDpsInterface>(*ctx); */
  });
}
