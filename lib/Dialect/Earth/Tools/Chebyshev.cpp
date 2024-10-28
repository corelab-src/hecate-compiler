#include "hecate/Dialect/Earth/Tools/Chebyshev.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

// ref :
// https://github.com/numpy/numpy/blob/v2.1.0/numpy/polynomial/chebyshev.py

using namespace hecate;

hecate::ChebyshevPoly::ChebyshevPoly(){};
hecate::ChebyshevPoly::ChebyshevPoly(const std::vector<double> &coeff)
    : coefficients(coeff) {}

std::vector<double>
hecate::ChebyshevPoly::chebyshev_to_zseries(const std::vector<double> &coeffs) {
  size_t n = coeffs.size();
  std::vector<double> zs(2 * n - 1, 0.0);
  for (size_t i = 0; i < n; i++) {
    zs[n - 1 + i] += coeffs[i] / 2.0;
    zs[n - 1 - i] += coeffs[i] / 2.0;
  }
  return zs;
}

std::vector<double>
ChebyshevPoly::zseries_to_chebyshev(const std::vector<double> &zs) {
  size_t n = (zs.size() + 1) / 2;
  std::vector<double> coeffs(n, 0.0);
  for (size_t i = 0; i < n; i++) {
    coeffs[i] = zs[n - 1 + i];
    if (i > 0) {
      coeffs[i] *= 2.0;
    }
  }
  return coeffs;
}

void hecate::ChebyshevPoly::poly_divide(const std::vector<double> &numerator,
                                        const std::vector<double> &denominator,
                                        std::vector<double> &quotient,
                                        std::vector<double> &remainder) {
  size_t lc1 = numerator.size();
  size_t lc2 = denominator.size();

  if (lc2 == 1) {
    quotient = numerator;
    for (auto &value : quotient) {
      value /= denominator[0];
    }
    remainder = {0.0};
    return;
  }

  if (lc1 < lc2) {
    quotient = {0.0};
    remainder = numerator;
    return;
  }

  std::vector<double> numerator_copy = numerator;
  std::vector<double> denominator_copy = denominator;
  size_t dlen = lc1 - lc2;
  double scl = denominator_copy[0];
  for (auto &value : denominator_copy) {
    value /= scl;
  }
  quotient.resize(dlen + 1, 0.0);

  size_t i = 0;
  size_t j = dlen;
  while (i <= j) {
    double r = numerator_copy[i];
    quotient[i] = r;
    quotient[dlen - i] = r;
    for (size_t k = 0; k < lc2; k++) {
      numerator_copy[i + k] -= r * denominator_copy[k];
      numerator_copy[j + k] -= r * denominator_copy[k];
    }
    i = i + 1;
    ;
    if (j > 0) {
      j = j - 1;
    } else {
      break;
    }
  }

  for (auto &val : quotient) {
    val /= scl;
  }

  remainder.assign(numerator_copy.begin() + i,
                   numerator_copy.begin() + i + lc2 - 1);
}

ChebyshevPoly
hecate::ChebyshevPoly::operator/(const ChebyshevPoly &denominator_cheby) {
  /*
  std::cout<<"numerator_cheby print"<<std::endl;
  print();
  std::cout<<"denominator_cheby
  print"<<denominator_cheby.coefficients.size()<<std::endl;
  denominator_cheby.print();
  */
  if (denominator_cheby.coefficients.empty()) {
    std::cerr << "Division by zero polynomial" << std::endl;
  }

  size_t lc1 = coefficients.size();
  size_t lc2 = denominator_cheby.coefficients.size();

  if (lc1 < lc2) {
    ChebyshevPoly quotient = ChebyshevPoly({0.0});
    return quotient;
  } else if (lc2 == 1) {
    std::vector<double> q = coefficients;
    for (auto &val : q) {
      val /= denominator_cheby.coefficients[0];
    }
    ChebyshevPoly quotient = ChebyshevPoly(q);
    return quotient;
  }

  std::vector<double> numerator = chebyshev_to_zseries(coefficients);
  std::vector<double> denominator =
      chebyshev_to_zseries(denominator_cheby.coefficients);

  std::vector<double> z_quotient;
  std::vector<double> z_remainder;
  poly_divide(numerator, denominator, z_quotient, z_remainder);

  std::vector<double> c_quotient = zseries_to_chebyshev(z_quotient);
  // std::vector<double> c_remainder = zseries_to_chebyshev(z_remainder);

  ChebyshevPoly quotient = ChebyshevPoly(c_quotient);
  // ChebyshevPoly remainder = ChebyshevPoly(c_remainder);

  // std::cout<<"numerator print"<<std::endl;
  // ChebyshevPoly(numerator).print();
  // std::cout<<"denominator print"<<std::endl;
  // ChebyshevPoly(denominator).print();
  // std::cout<<"quotient print"<<std::endl;
  // ChebyshevPoly(z_quotient).print();
  std::cout << "quotient_cheby print" << std::endl;
  quotient.print();
  // std::cout<<"remainder print"<<std::endl;
  // ChebyshevPoly(z_remainder).print();
  // std::cout<<"remainder_cheby print"<<std::endl;
  // remainder.print();

  return quotient;
}

ChebyshevPoly
hecate::ChebyshevPoly::operator%(const ChebyshevPoly &denominator_cheby) {
  if (denominator_cheby.coefficients.empty()) {
    std::cerr << "Division by zero polynomial" << std::endl;
  }

  size_t lc1 = coefficients.size();
  size_t lc2 = denominator_cheby.coefficients.size();

  if (lc1 < lc2) {
    ChebyshevPoly remainder = ChebyshevPoly(coefficients);
    return remainder;
  } else if (lc2 == 1) {
    ChebyshevPoly remainder = ChebyshevPoly({0.0});
    return remainder;
  }

  std::vector<double> numerator = chebyshev_to_zseries(coefficients);
  std::vector<double> denominator =
      chebyshev_to_zseries(denominator_cheby.coefficients);

  std::vector<double> z_quotient;
  std::vector<double> z_remainder;
  poly_divide(numerator, denominator, z_quotient, z_remainder);

  std::vector<double> c_remainder = zseries_to_chebyshev(z_remainder);

  ChebyshevPoly remainder = ChebyshevPoly(c_remainder);

  return remainder;
}

int hecate::ChebyshevPoly::coeff_size() { return coefficients.size(); }

double hecate::ChebyshevPoly::nth_coeff(int n) {
  if (n < coefficients.size()) {
    return coefficients[n];
  } else {
    return 0.0;
  }
}

void hecate::ChebyshevPoly::print() const {
  for (const auto &c : coefficients) {
    std::cout << c << " ";
  }
  std::cout << std::endl;
}
