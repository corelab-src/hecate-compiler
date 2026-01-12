#ifndef FRONTEND_H
#define FRONTEND_H
#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexOps.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <nlohmann/json.hpp>

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"
#include <vector>

using namespace mlir;
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
  llvm::SmallVector<mlir::func::FuncOp, 32> funcMap;
};

inline Context::Context() : ctxt(), mod(), builder() {

  ctxt.getOrLoadDialect<hecate::earth::EarthDialect>();
  ctxt.getOrLoadDialect<mlir::func::FuncDialect>();
  ctxt.getOrLoadDialect<scf::SCFDialect>();
  ctxt.getOrLoadDialect<affine::AffineDialect>();
  ctxt.getOrLoadDialect<index::IndexDialect>();
  ctxt.getOrLoadDialect<transform::TransformDialect>();
  ctxt.getOrLoadDialect<arith::ArithDialect>();
  ctxt.getOrLoadDialect<shape::ShapeDialect>();

  registry.applyExtensions(&ctxt);

  auto tmp = std::make_unique<mlir::OpBuilder>(&ctxt);
  builder.swap(tmp);
  auto ttmp = std::make_unique<mlir::IRRewriter>(*builder);
  rewriter.swap(ttmp);

  mod = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder->getUnknownLoc()));
}
// COMMON
//  std::vector<int> get2DParam(const nlohmann::json &j, const std::string &key,
//  int defaultVal); mlir::Value createPaddedInput(mlir::OpBuilder &builder,
//  mlir::Location loc,
//                          mlir::Value input, mlir::ArrayRef<int> padding);
//
//
//  //Tensor
//  mlir::Value createBatchNorm2dOp(mlir::OpBuilder &builder, mlir::Location
//  loc,
//                             mlir::Value input, mlir::Value BN_G, mlir::Value
//                             BN_H, const nlohmann::json &parsed_args);
//  mlir::Value createConv2dOp(mlir::OpBuilder &builder, mlir::Location loc,
//                             mlir::Value input, mlir::Value kernel,
//                             const nlohmann::json &parsed_args);
//
//  mlir::Value createAvgPool2dOp(mlir::OpBuilder &builder, mlir::Location loc,
//                             mlir::Value input, const nlohmann::json
//                             &parsed_args);
//  mlir::Value createSiLUOp(mlir::OpBuilder &builder, mlir::Location loc,
//                             mlir::Value input, const nlohmann::json
//                             &parsed_args);
//
//  extern "C" {
//  Context *init();
//  void finalize(Context *ctxt);
//  }

} // namespace hecate

#endif // FRONTEND_H
