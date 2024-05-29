
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// ForOpInterface
//===----------------------------------------------------------------------===//

struct ForOpDpsInterface
    : public mlir::DestinationStyleOpInterface::ExternalModel<
          ForOpDpsInterface, mlir::scf::ForOp> {

  ::mlir::MutableOperandRange getDpsInitsMutable(Operation *op) const {
    return {op, 0, 1};
  }
};

struct ForOpHEVMInterface
    : public hecate::ckks::HEVMOpInterface::ExternalModel<ForOpHEVMInterface,
                                                          mlir::scf::ForOp> {

  HEVMOperation
  getHEVMOperation(Operation *op, llvm::DenseMap<mlir::Value, int64_t> plainMap,
                   llvm::DenseMap<mlir::Value, int64_t> cipherMap,
                   llvm::DenseMap<mlir::Value, int64_t> integerMap) const {
    HEVMOperation heop;
    heop.opcode = 11;
    heop.dst = integerMap[op->getOperand(0)];
    heop.lhs = integerMap[op->getOperand(1)];
    heop.rhs = integerMap[op->getOperand(2)];
    return heop;
  }
};

//===----------------------------------------------------------------------===//
// YieldOpInterface
//===----------------------------------------------------------------------===//

struct YieldOpDpsInterface
    : public mlir::DestinationStyleOpInterface::ExternalModel<
          YieldOpDpsInterface, mlir::scf::YieldOp> {
  ::mlir::MutableOperandRange getDpsInitsMutable(Operation *op) const {
    return {op, 0, 1};
  }
};

struct YieldOpHEVMInterface
    : public hecate::ckks::HEVMOpInterface::ExternalModel<YieldOpHEVMInterface,
                                                          mlir::scf::YieldOp> {

  HEVMOperation
  getHEVMOperation(Operation *op, llvm::DenseMap<mlir::Value, int64_t> plainMap,
                   llvm::DenseMap<mlir::Value, int64_t> cipherMap,
                   llvm::DenseMap<mlir::Value, int64_t> integerMap) const {
    HEVMOperation heop;
    heop.opcode = 12;
    heop.dst = 0;
    heop.lhs = 0;
    heop.rhs = 0;
    return heop;
  }
};

void hecate::ckks::registerSCFOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.insert<scf::SCFDialect>();
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    /* scf::ForOp::attachInterface<ForOpDpsInterface>(*ctx); */
    /* scf::YieldOp::attachInterface<YieldOpDpsInterface>(*ctx); */
    scf::ForOp::attachInterface<ForOpHEVMInterface>(*ctx);
    scf::YieldOp::attachInterface<YieldOpHEVMInterface>(*ctx);
  });
}
