
#ifndef HECATE_TOOLS_GENPOLY
#define HECATE_TOOLS_GENPOLY

#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <vector>
#include <string>

namespace hecate {

struct PolynomialAnalysis {
public:
  PolynomialAnalysis();
  
  int64_t GenPoly_Test(int degree = 16);
  int64_t GenPoly(const std::vector<std::string> &tree_var,
                         const std::vector<std::string> &coeff_var,
                         int degree,
                         float scale = 1.0);
  std::vector<std::string> LoadVar(const std::string &filename);
  //int64_t Chebyshev(std::vector<double> coeff);
  //std::vector<std::tuple<int,int,int,int>> Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff);
  int64_t Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff);
  //int64_t GSBS(mlir::RankedTensorType input);
  int64_t GSBS();
private:
  int length;

};
} // namespace hecate

#endif
