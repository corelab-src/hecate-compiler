
#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <memory>

#include "frontend.h"
#include "hecate/Support/Support.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/SourceMgr.h"
#include <execinfo.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace mlir;

void handler(int sig) {
  void *array[10];
  size_t size;

  size = backtrace(array, 10);

  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

namespace hecate {

extern "C" {

valueID createConstant(Context *ctxt, double *data, int64_t len, char *filename,
                       size_t line) {
  auto &&builder = *ctxt->builder;
  auto cons = builder.create<hecate::earth::ConstantOp>(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0),
      llvm::ArrayRef(data, len));

  ctxt->valueMap.push_back(cons);
  return ctxt->valueMap.size() - 1;
}
valueID createArithConstant(Context *ctxt, int data, char *filename,
                            size_t line) {
  auto &&builder = *ctxt->builder;
  auto cons = builder.create<mlir::arith::ConstantIndexOp>(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0),
      data);
  ctxt->valueMap.push_back(cons);
  return ctxt->valueMap.size() - 1;
}
// funcID createFunc(Context *ctxt, char *name, int *inputTys, size_t len,
// char *filename, size_t line) {
funcID createFunc(Context *ctxt, char *name, char **inputTys, size_t len,
                  char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&funcMap = ctxt->funcMap;
  llvm::SmallVector<mlir::Type, 4> arg_types;
  for (size_t i = 0; i < len; i++) {
    if (*inputTys[i] == 'c')
      arg_types.push_back(mlir::RankedTensorType::get(
          llvm::SmallVector<int64_t, 1>{1},
          builder.getType<hecate::earth::CipherType>(0, 0)));
    else if (*inputTys[i] == 'i') {
      arg_types.push_back(mlir::IndexType::get(&ctxt->ctxt));
    }
  }
  auto funcType = builder.getFunctionType(
      arg_types, mlir::RankedTensorType::get(
                     llvm::SmallVector<int64_t, 1>{1},
                     builder.getType<hecate::earth::CipherType>(0, 0)));
  auto funcOp = mlir::func::FuncOp::create(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0),
      std::string("_hecate_") + name, funcType);
  funcMap.push_back(funcOp);
  ctxt->mod->push_back(funcOp);
  return funcMap.size() - 1;
}

void initFunc(Context *ctxt, funcID fun, valueID *args, size_t len) {
  auto &&funcOp = ctxt->funcMap[fun];
  auto &&valueMap = ctxt->valueMap;
  auto entryBlock = funcOp.addEntryBlock();
  auto funcInput = entryBlock->getArguments();
  ctxt->builder->setInsertionPointToStart(entryBlock);
  {
    int i = 0;
    for (auto a : funcInput) {
      valueMap.push_back(a);
      args[i++] = valueMap.size() - 1;
    }
  }
}

char *save(Context *c, char *const_name, char *mlir_name) {
  c->mod->getOperation()->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                                  c->builder->getStringAttr(mlir_name));
  std::string s_const_name(const_name);
  mlir::PassManager pm(&c->ctxt);
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(hecate::earth::createLoopRotation());
  /* pm.addNestedPass<func::FuncOp>( */
  /*     earth::createElideConstant({s_const_name + "/"})); */
  pm.addNestedPass<func::FuncOp>(earth::createPrivatizeConstant());
  pm.addPass(createCanonicalizerPass());

  auto ret = pm.run(*c->mod);
  std::error_code EC;
  llvm::raw_fd_ostream outputFile(mlir_name, EC);
  c->mod->print(outputFile, mlir::OpPrintingFlags()
                                .printGenericOpForm()
                                .enableDebugInfo()
                                .useLocalScope());
  c->valueMap.clear();
  c->funcMap.clear();
  c->mod.release();
  return mlir_name;
}

/* Unary Operation */
valueID createUnary(Context *ctxt, size_t opcode, valueID lhs, char *filename,
                    size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&valueMap = ctxt->valueMap;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  auto &&source = valueMap[lhs];
  switch (opcode) {
  case 0: {
    auto res = builder.create<hecate::earth::BootstrapOp>(location, source);
    valueMap.push_back(res);
    break;
  }
  case 13: {
    if (source.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      auto res = builder.create<hecate::earth::NegateOp>(location, source);
      valueMap.push_back(res);
    }
    break;
  }
  default:
    assert(0 && "Unary Operation type is wrong");
  }
  return valueMap.size() - 1;
}

/* Binary Operation */
valueID createBinary(Context *ctxt, size_t opcode, valueID lhs, valueID rhs,
                     char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&valueMap = ctxt->valueMap;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);

  auto &&srcl = valueMap[lhs];
  auto &&srcr = valueMap[rhs];

  switch (opcode) {
  case 6: {
    auto res = builder.create<hecate::earth::AddOp>(location, srcl, srcr);
    valueMap.push_back(res);
    break;
  }
  case 7: {
    if (srcr.getType().dyn_cast<hecate::earth::HEScaleTypeInterface>()) {
      auto neg = builder.create<hecate::earth::NegateOp>(location, srcr);
      valueMap.push_back(neg);
      auto res = builder.create<hecate::earth::AddOp>(location, srcl, neg);
      valueMap.push_back(res);
    } else {
      auto res = builder.create<mlir::arith::SubIOp>(location, srcl, srcr);
      valueMap.push_back(res);
    }
    break;
  }

  case 8: {
    auto res = builder.create<hecate::earth::MulOp>(location, srcl, srcr);
    valueMap.push_back(res);
    break;
  }
  case 101: {
    auto res = builder.create<mlir::index::ShlOp>(location, srcl, srcr);
    valueMap.push_back(res);
    break;
  }

  default:
    assert(0 && "Binary Operation type is wrong");
  }
  return valueMap.size() - 1;
}

valueID createRotation(Context *ctxt, size_t valueID, int offset,
                       char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  auto &&srcl = ctxt->valueMap[valueID];
  auto cons = builder.create<hecate::earth::RotateOp>(location, srcl, offset);
  ctxt->valueMap.push_back(cons);
  return ctxt->valueMap.size() - 1;
}

void mlir_dump(Context *ctxt) { ctxt->mod->dump(); }

void createCall(Context *ctxt, funcID fid, valueID *args, valueID *rets,
                size_t len, char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);

  auto &&callee = ctxt->funcMap[fid];
  MLIRContext *ctx = callee.getContext();
  mlir::OpBuilder::InsertPoint ip = builder.saveInsertionPoint();
  // Get existing 'call_count' attribute (if any)
  IntegerAttr countAttr = callee->getAttrOfType<IntegerAttr>("call_count");
  // If the attribute exists, increment it; otherwise start from 1
  int64_t newCount = 1;
  if (countAttr)
    newCount = countAttr.getInt() + 1;

  // Create a new i64 IntegerAttr with the updated count
  auto callCountAttr = IntegerAttr::get(IntegerType::get(ctx, 64), newCount);

  // Set (or overwrite) the 'call_count' attribute on the callee
  callee->setAttr("call_count", callCountAttr);

  // Prepare types
  mlir::Type cipherTy =
      RankedTensorType::get(llvm::SmallVector<int64_t, 1>{1},
                            builder.getType<hecate::earth::CipherType>(0, 0));
  mlir::Type erasedTy =
      RankedTensorType::get(llvm::SmallVector<int64_t, 1>{1},
                            builder.getType<hecate::earth::ErasedType>());

  // callee-side
  //  First time calling this function, set types as ErasedType/CipherType
  if (!countAttr) {
    // set function type ErasedType -> ErasedType
    llvm::SmallVector<mlir::Type, 1> argTypes(callee.getNumArguments(),
                                              erasedTy);
    llvm::SmallVector<mlir::Type, 1> retTypes(callee.getNumResults(), erasedTy);
    callee.setType(builder.getFunctionType(argTypes, retTypes));

    // set argument types in function block
    mlir::Block &entryBlock = callee.getBody().front();
    for (auto arg : entryBlock.getArguments()) {
      arg.setType(erasedTy);
    }

    // Arguments in function set as CipherType with level 0
    builder.setInsertionPointToStart(&callee.front());
    for (auto arg : callee.getArguments()) {
      auto castOp =
          builder.create<hecate::earth::LevelCastOp>(location, cipherTy, arg);
      arg.replaceAllUsesExcept(castOp.getResult(), castOp);
    }
    // Results in function set as ErasedType
    auto terminator = callee.front().getTerminator();
    for (size_t i = 0; i < terminator->getNumOperands(); i++) {
      builder.setInsertionPoint(callee.front().getTerminator());
      auto castOp = builder.create<hecate::earth::LevelCastOp>(
          location, erasedTy, terminator->getOperand(i));
      terminator->setOperand(i, castOp);
    }
  }
  builder.restoreInsertionPoint(ip);
  // caller-side
  // inputs set as ErasedType
  llvm::SmallVector<mlir::Value> inputs;
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[args[i]];
    inputs.push_back(v);
    // auto castOp =
    // builder.create<hecate::earth::LevelCastOp>(location, erasedTy, v);
    // inputs.push_back(castOp);
  }
  auto callOp = builder.create<mlir::func::CallOp>(
      location, callee.getFunctionType().getResults(), callee.getName(),
      inputs);
  builder.setInsertionPoint(callOp);
  for (size_t i = 0; i < callOp.getNumOperands(); i++) {
    auto v = callOp.getOperand(i);
    auto castOp =
        builder.create<hecate::earth::LevelCastOp>(location, erasedTy, v);
    callOp.setOperand(i, castOp);
  }

  //  Results set as ErasedType
  builder.setInsertionPointAfter(callOp);
  for (size_t i = 0; i < callOp.getResults().size(); i++) {
    auto v = callOp.getResult(i);
    auto castOp =
        builder.create<hecate::earth::LevelCastOp>(location, cipherTy, v);
    ctxt->valueMap.push_back(castOp);
    rets[i] = ctxt->valueMap.size() - 1;
  }
}

void setOutput(Context *ctxt, funcID fun, valueID *ret, size_t len) {
  llvm::SmallVector<mlir::Value, 2> rets;
  llvm::SmallVector<mlir::Type, 1> types;
  for (int i = 0; i < len; i++) {
    rets.push_back(ctxt->valueMap[ret[i]]);
    types.push_back(ctxt->valueMap[ret[i]].getType());
  }
  auto func = ctxt->funcMap[fun];
  ctxt->builder->setInsertionPointToEnd(&func.front());
  ctxt->builder->create<mlir::func::ReturnOp>(func.getLoc(), rets);
  auto retType = func.getFunctionType();
  func.setFunctionType(
      ctxt->builder->getFunctionType(retType.getInputs(), types));
}

Context *init() {
  /* signal(SIGSEGV, handler); */
  return new ::hecate::Context();
}
void finalize(Context *ctxt) { delete ctxt; }
} // namespace hecate
} // namespace hecate

/* int main() {} */
