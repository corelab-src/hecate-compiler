
#ifndef HECATE_TOOLS_GENPOLY
#define HECATE_TOOLS_GENPOLY

#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "hecate/Dialect/Earth/Tools/Chebyshev.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <vector>
#include <string>

namespace hecate {

struct GenPoly {
public:
  GenPoly();
  
  GenPoly(int degree);
  GenPoly(const std::string &treeStr,
                         const std::string &coeffStr,
                         int degree,
                         float scale_in);
  GenPoly(const std::vector<std::string> &treeVar,
                         const std::vector<std::string> &coeffVar,
                         int degree,
                         float scale_in);
  std::vector<std::string> LoadVar(const std::string &filename);
  void GenPoly_run();

  void Calc_Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff);
  std::vector<double> GSBS_check(std::vector<double> input);
  //int64_t GSBS_check();
  //int64_t GSBS_createHEOps(mlir::RankedTensorType input);
  //int64_t GSBS_createHEOps();
  //void GSBS_createHEOps(mlir::OpBuilder &builder, mlir::Location loc, hecate::earth::CipherType input);
  mlir::Value GSBS_createHEOps(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value input);
private:
  int length = 16;
  float scale = 1.0;
  std::vector<std::string> tree_var;
  std::vector<std::string> coeff_var;
  std::vector<std::tuple<int, int, int, hecate::ChebyshevPoly>> calc_order;
};
} // namespace hecate

#endif
