
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
  
  std::vector<std::vector<double>> firstkind(int n);
  std::vector<double> secondkind(int n);
  
  std::vector<double> chebyshev_to_monomial(const std::vector<double>& coeffs);
  std::vector<double> monomial_to_chebyshev(const std::vector<double>& monCoeffs, int degree);
  void poly_divide(const std::vector<double>& numerator,
                   const std::vector<double>& denominator,
                   std::vector<double>& quotient,
                   std::vector<double>& remainder);

  //ChebyshevPoly divide_quotient(const ChebyshevPoly& denominator_cheby);
  //ChebyshevPoly divide_remainder(const ChebyshevPoly& denominator_cheby);
  ChebyshevPoly operator/(const ChebyshevPoly& denominator_cheby);
  ChebyshevPoly operator%(const ChebyshevPoly& denominator_cheby);
  
  void print() const;
  int coeff_size();
  double nth_coeff(int n);
private:
  std::vector<double> coefficients;
};
} // namespace hecate

#endif
