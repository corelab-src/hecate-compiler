
#ifndef HECATE_TOOLS_CHEBYSHEV
#define HECATE_TOOLS_CHEBYSHEV

#include <iostream>
#include <string>
#include <vector>

namespace hecate {

struct ChebyshevPoly {
public:
  ChebyshevPoly();
  ChebyshevPoly(const std::vector<double> &coeff);

  std::vector<double> chebyshev_to_zseries(const std::vector<double> &coeffs);
  std::vector<double> zseries_to_chebyshev(const std::vector<double> &zs);

  void poly_divide(const std::vector<double> &numerator,
                   const std::vector<double> &denominator,
                   std::vector<double> &quotient,
                   std::vector<double> &remainder);

  ChebyshevPoly operator/(const ChebyshevPoly &denominator_cheby);
  ChebyshevPoly operator%(const ChebyshevPoly &denominator_cheby);

  void print() const;
  int coeff_size();
  double nth_coeff(int n);

private:
  std::vector<double> coefficients;
};
} // namespace hecate

#endif
