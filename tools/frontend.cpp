
#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <memory>

#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"

#include <execinfo.h>
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

using valueID = size_t;
using loopID = size_t;
using funcID = size_t;

struct Context {
  Context();
  mlir::MLIRContext ctxt;
  DialectRegistry registry;
  mlir::OwningOpRef<mlir::ModuleOp> mod;
  std::unique_ptr<mlir::OpBuilder> builder;
  std::unique_ptr<mlir::IRRewriter> rewriter;
  llvm::SmallVector<mlir::Value, 32> valueMap;
  llvm::SmallVector<mlir::scf::ForOp, 32> loopMap;
  /* llvm::SmallVector<affine::AffineForOp, 32> loopMap; */
  llvm::SmallVector<mlir::func::FuncOp, 32> funcMap;
};

Context::Context() : ctxt(), mod(), builder() {

  ctxt.getOrLoadDialect<hecate::earth::EarthDialect>();
  ctxt.getOrLoadDialect<mlir::func::FuncDialect>();
  ctxt.getOrLoadDialect<scf::SCFDialect>();
  ctxt.getOrLoadDialect<affine::AffineDialect>();
  ctxt.getOrLoadDialect<index::IndexDialect>();
  ctxt.getOrLoadDialect<transform::TransformDialect>();
  ctxt.getOrLoadDialect<arith::ArithDialect>();

  registry.applyExtensions(&ctxt);

  auto tmp = std::make_unique<mlir::OpBuilder>(&ctxt);
  builder.swap(tmp);
  auto ttmp = std::make_unique<mlir::IRRewriter>(*builder);
  rewriter.swap(ttmp);

  mod = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder->getUnknownLoc()));
}

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
funcID createFunc(Context *ctxt, char *name, int *inputTys, size_t len,
                  char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&funcMap = ctxt->funcMap;
  llvm::SmallVector<mlir::Type, 4> arg_types(len);
  std::transform(inputTys, inputTys + len, arg_types.begin(), [&](auto a) {
    return mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t, 1>{1},
        builder.getType<hecate::earth::CipherType>(0, 0));
  });
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
  c->mod->dump();
  std::string s_const_name(const_name);
  mlir::PassManager pm(&c->ctxt);
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      earth::createElideConstant({s_const_name + "/"}));
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

loopID createLoop(Context *ctxt, size_t *rng, valueID *indvar, valueID *inputs,
                  size_t len, char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  llvm::SmallVector<mlir::Value> inputarr;
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[inputs[i]];
    auto er = builder.create<hecate::earth::EraseTypeOp>(
        location, ctxt->valueMap[inputs[i]]);
    inputarr.push_back(er);
    ctxt->valueMap.push_back(er);
  }

  /* auto loop = builder.create<affine::AffineForOp>(location, rng[0], rng[1],
   */
  /*                                                 rng[2], inputarr); */
  auto lowerBound =
      builder.create<mlir::arith::ConstantIndexOp>(location, rng[0]);
  auto upperBound =
      builder.create<mlir::arith::ConstantIndexOp>(location, rng[1]);
  auto step = builder.create<mlir::arith::ConstantIndexOp>(location, rng[2]);
  auto loop = builder.create<mlir::scf::ForOp>(location, lowerBound, upperBound,
                                               step, inputarr);
  ctxt->loopMap.push_back(loop);

  builder.setInsertionPointToStart(loop.getBody());
  ctxt->valueMap.push_back(loop.getInductionVar());
  indvar[0] = ctxt->valueMap.size() - 1;

  mlir::Region &loop_body = loop.getBodyRegion();
  mlir::Block &loop_block = loop_body.front();
  for (size_t i = 0; i < len; i++) {
    auto carriedVar = loop_block.getArgument(i + 1);
    ctxt->valueMap.push_back(carriedVar);
    indvar[i + 1] = ctxt->valueMap.size() - 1;
  }
  return ctxt->loopMap.size() - 1;
}

valueID getInductionVar(Context *ctxt, loopID loopID) {
  auto &&loop = ctxt->loopMap[loopID];
  Value iv = loop.getInductionVar();
  ctxt->valueMap.push_back(iv);
  return ctxt->valueMap.size() - 1;
}

void setLoopCarriedVars(Context *ctxt, valueID loopID, valueID *arg, size_t len,
                        char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&rewriter = *ctxt->rewriter;
  auto &&loop = ctxt->loopMap[loopID];
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  mlir::Region &loop_body = loop.getBodyRegion();
  mlir::Block &loop_block = loop_body.front();

  // check the type conflict

  // set loop argument
  SmallPtrSet<mlir::Operation *, 2> excepted;
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[arg[i]];
    auto carriedVar = loop_block.getArgument(i + 1);
    for (auto use : v.getUsers()) {
      if (loop.isDefinedOutsideOfLoop(use->getResult(0))) {
        excepted.insert(use);
      }
    }
    v.replaceAllUsesExcept(carriedVar, excepted);
  }
}

void setYield(Context *ctxt, valueID loopID, valueID *ret, size_t len,
              char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&loop = ctxt->loopMap[loopID];
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  mlir::Region &loop_body = loop.getBodyRegion();
  mlir::Block &loop_block = loop_body.front();
  llvm::SmallVector<mlir::Value> rets;
  Value iv = loop.getInductionVar();
  for (size_t i = 0; i < len; i++) {
    auto v = ctxt->valueMap[ret[i]];
    auto er = builder.create<hecate::earth::EraseTypeOp>(location, v);
    rets.push_back(er);
  }
  auto loopYield = builder.create<mlir::scf::YieldOp>(location, rets);
  for (size_t i = 0; i < len; i++) {
    ctxt->valueMap.push_back(loop->getResult(i));
    ret[i] = ctxt->valueMap.size() - 1;
  }
  builder.setInsertionPointToEnd(loop->getBlock());
}

void setOutput(Context *ctxt, funcID fun, valueID *ret, size_t len) {
  llvm::SmallVector<mlir::Value, 2> rets;
  llvm::SmallVector<mlir::Type, 1> types;
  for (int i = 0; i < len; i++) {
    rets.push_back(ctxt->valueMap[ret[i]]);
    types.push_back(ctxt->valueMap[ret[i]].getType());
  }
  auto func = ctxt->funcMap[fun];
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
