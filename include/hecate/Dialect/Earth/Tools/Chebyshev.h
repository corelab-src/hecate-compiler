
#ifndef HECATE_TOOLS_CHEBYSHEV
#define HECATE_TOOLS_CHEBYSHEV

#include "hecate/Dialect/Earth/IR/HEParameterInterface.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <vector>
#include <string>

namespace hecate {

struct ChebyshevPoly {
public:
  ChebyshevPoly(const std::vector<double>& coeff);
  
  double evaluate(double x) const;
  ChebyshevPoly divide_quotient(const ChebyshevPoly& denominator_cheby);
  ChebyshevPoly divide_remainder(const ChebyshevPoly& denominator_cheby);
  void print() const;
private:
  std::vector<double> coefficients;
};
} // namespace hecate

#endif
