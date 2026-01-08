#ifndef HECATE_TOOLS_HECATEOPT_HECATEOPTMAIN_H
#define HECATE_TOOLS_HECATEOPT_HECATEOPTMAIN_H

#include "hecate/Conversion/CKKSToCKKS/ArithToDpsArith.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>
#include <iostream>

#include "hecate/Conversion/Passes.h"
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"

using namespace llvm;

using namespace mlir;
using namespace hecate;

void registerHecatePipeline(cl::opt<std::string> &outputFilename);
// void registerPrototypePipeline(cl::opt<std::string> &outputFilename);
struct HecateOptOptions {
  // --- Scale Management Options ---
  cl::opt<int64_t> waterline{
      "waterline", cl::desc("Waterline of scale management"), cl::init(20)};

  cl::opt<int64_t> output_val{
      "output-val", cl::desc("Output value upper bound of scale management"),
      cl::init(10)};

  // --- Debug & Tool Options ---
  cl::opt<bool> enable_printer{
      "enable-debug-printer",
      cl::desc(
          "Enable printing after scale management and earth-ckks conversion"),
      cl::init(false)};

  cl::opt<bool> enable_check_smu{
      "enable-check-smu", cl::desc("Check SMU generation"), cl::init(false)};

  // --- ELASM Optimization Options ---
  cl::opt<int64_t> parallel_elasm{
      "parallel-elasm", cl::desc("Parallel # of ELASM"), cl::init(20)};

  cl::opt<int64_t> num_iter_elasm{"num-iter-elasm",
                                  cl::desc("# Iters of ELASM"), cl::init(1000)};

  cl::opt<int64_t> beta_elasm{"beta-elasm", cl::desc("Beta of ELASM"),
                              cl::init(70)};

  cl::opt<int64_t> gamma_elasm{"gamma-elasm", cl::desc("Gamma of ELASM"),
                               cl::init(50)};

  // --- DaCapo Optimization Options ---
  cl::opt<float> threshold{
      "threshold",
      cl::desc("Scale coverage threshold of DaCapo, 0 <= threshold <= 1"),
      cl::init(0.5)};

  // --- HALO Optimization Options ---
  cl::opt<int64_t> unroll_factor{"unroll-factor",
                                 cl::desc("Unroll factor for scale management"),
                                 cl::init(1)};
};
#endif // HECATE_TOOLS_HECATEOPT_HECATEOPTMAIN_H
