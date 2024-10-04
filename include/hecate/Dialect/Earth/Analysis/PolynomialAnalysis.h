
#ifndef HECATE_ANALYSIS_POLYNOMIALANALYSIS
#define HECATE_ANALYSIS_POLYNOMIALANALYSIS

#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/AnalysisManager.h"

#include <vector>
#include <string>

namespace hecate {

struct PolynomialAnalysis {
public:
  PolynomialAnalysis(mlir::Operation *op);
  
  int64_t GenPoly_Test(int degree = 16);
  int64_t GenPoly(const std::vector<std::string> &tree_var,
                         const std::vector<std::string> &coeff_var,
                         int degree,
                         float scale = 1.0);
  std::vector<std::string> LoadVar(const std::string &filename);
  //int64_t Chebyshev(std::vector<double> coeff);
  //std::vector<std::tuple<int,int,int,int>> Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff);
  int64_t Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff);
  int64_t GSBS();
private:
  mlir::Operation *_op;

};
} // namespace hecate

#endif
