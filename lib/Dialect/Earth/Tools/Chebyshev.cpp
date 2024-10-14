#include "hecate/Dialect/Earth/Tools/Chebyshev.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "mlir/IR/BuiltinTypes.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cmath>
#include <tuple>
#include <map>
#include <algorithm>
#include <variant>

using namespace hecate;

hecate::ChebyshevPoly::ChebyshevPoly(const std::vector<double>& coeff) : coefficients(coeff) {}

std::vector<std::vector<double>> hecate::ChebyshevPoly::firstkind(int n) {
  /* If the model uses chebyshev calculation many times, 
     it is better to save a answer result */
  std::vector<std::vector<double>> Tn(n+1);
  // T0(x) = 1
  Tn[0]={1};
  if(n==0) {
    return Tn;
  }
  Tn[1]={0, 1}; // T1(x)=x
       
  for (int k=2; k<=n; k++) {
    std::vector<double> Tk(Tn[k-1].size()+1, 0.0);
    // Tn(x) = 2 * x * T(n-1)(x) - T(n-2)(x)
    for (size_t j=0;j<Tn[k-1].size(); j++) {
      Tk[j+1] += 2 * Tn[k-1][j];
    }
    for (size_t j=0; j<Tn[k-2].size();j++) {
      Tk[j] -= Tn[k-2][j];
    }
    Tn[k] = Tk;
  }  
  return Tn;
}

std::vector<double> hecate::ChebyshevPoly::secondkind(int n) {
  std::vector<double> Un(n+1, 0.0);
  // U0(x) = 1
  if(n==0) {
    Un[0]=1;
  }
  else if(n==1) {
    Un[0]=0; 
    Un[1]=2; // U1(x)=2x
  }
  else {
    std::vector<double> U0 = {1};
    std::vector<double> U1 = {0, 2};
    for (int k=2; k<=n; k++) {
      std::vector<double> Uk(k+1, 0.0);
      // Un(x) = 2 * x * U(n-1)(x) - U(n-2)(x)
      for (size_t j=0;j<U1.size(); j++) {
        Uk[j+1] += 2 * U1[j];
      }
      for (size_t j=0; j<U0.size();j++) {
        Uk[j] -= U0[j];
      }
      U0 = U1;
      U1 = Uk;
      Un = Uk;
    }
  }
  return Un;
}

std::vector<double> hecate::ChebyshevPoly::chebyshev_to_monomial(const std::vector<double>& coeffs) {
  int N = coeffs.size()-1;
  std::vector<std::vector<double>> monomial_basis = firstkind(N);
  std::vector<double> monomial_coeffs(monomial_basis[N].size(), 0.0);

  for (int n=0; n<=N; n++) {
    double c = coeffs[n];
    const std::vector<double>& Tn = monomial_basis[n];
    for (size_t k=0; k<Tn.size(); k++) {
      monomial_coeffs[k] += c * Tn[k];
    }
  }
  return monomial_coeffs;
}



std::vector<double> hecate::ChebyshevPoly::monomial_to_chebyshev(const std::vector<double>& monCoeffs, int degree) {
    const int N = degree;
    const int numPoints = 2 * N + 1; // Number of sample points
    std::vector<double> chebCoeffs(N + 1, 0.0);

    // Chebyshev nodes
    std::vector<double> x(numPoints);
    for (int k = 0; k < numPoints; ++k) {
        x[k] = std::cos(M_PI * (k + 0.5) / numPoints);
    }

    // Evaluate monomial polynomial at Chebyshev nodes
    std::vector<double> y(numPoints, 0.0);
    for (int k = 0; k < numPoints; ++k) {
        double xi = x[k];
        double xi_pow = 1.0;
        for (size_t i = 0; i < monCoeffs.size(); ++i) {
            y[k] += monCoeffs[i] * xi_pow;
            xi_pow *= xi;
        }
    }

    // Compute Chebyshev coefficients using the discrete orthogonality
    for (int n = 0; n <= N; ++n) {
        double sum = 0.0;
        for (int k = 0; k < numPoints; ++k) {
            sum += y[k] * std::cos(n * M_PI * (k + 0.5) / numPoints);
        }
        chebCoeffs[n] = (2.0 / numPoints) * sum;
    }
    chebCoeffs[0] *= 0.5; // Adjust the first coefficient

    return chebCoeffs;
}



void hecate::ChebyshevPoly::poly_divide(const std::vector<double>& numerator, 
                                        const std::vector<double>& denominator,
                                        std::vector<double>& quotient,
                                        std::vector<double>& remainder) {
  std::vector<double> num = numerator;
  std::vector<double> den = denominator;

  int n = num.size() - 1;
  int m = den.size() - 1;

  if (m < 0) {
    std::cerr<<"Division by zero polynomial"<<std::endl;
  }

  quotient.assign(std::max(0, n-m + 1), 0.0);
  for (int i = n-m; i >= 0; --i) {
    quotient[i] = num[m+i]/den[m];
    for (int j = m+i; j >= i; --j) {
      num[j] -= quotient[i]*den[j-i];
    }
  }
    
  // The remainder is the lower-degree part of num
  remainder.assign(num.begin(), num.begin()+m);
}

//ChebyshevPoly hecate::ChebyshevPoly::divide_quotient(const ChebyshevPoly& denominator_cheby) {
ChebyshevPoly hecate::ChebyshevPoly::operator/(const ChebyshevPoly& denominator_cheby) {
    /* wrong divide!!! must change for chebyshev ******/
  
  /*
  std::cout<<"numerator_cheby print"<<std::endl;
  print();
  std::cout<<"denominator_cheby print"<<denominator_cheby.coefficients.size()<<std::endl;
  denominator_cheby.print();
  */

  std::vector<double> numerator = chebyshev_to_monomial(coefficients);
  std::vector<double> denominator = chebyshev_to_monomial(denominator_cheby.coefficients);

  std::vector<double> quotient;
  std::vector<double> remainder;
  poly_divide(numerator, denominator, quotient, remainder);
  
  // Convert quotient back to Chebyshev
  ChebyshevPoly quotient_value = ChebyshevPoly(monomial_to_chebyshev(quotient, quotient.size()-1));
  //ChebyshevPoly remainder_value = ChebyshevPoly(monomial_to_chebyshev(remainder, remainder.size()-1));

  /*
  std::cout<<"numerator print"<<std::endl;
  ChebyshevPoly(numerator).print();
  std::cout<<"denominator print"<<std::endl;
  ChebyshevPoly(denominator).print();
  std::cout<<"quotient print"<<std::endl;
  ChebyshevPoly(quotient).print();
  std::cout<<"quotient_cheby print"<<std::endl;
  quotient_value.print();
  std::cout<<"remainder print"<<std::endl;
  ChebyshevPoly(remainder).print();
  std::cout<<"remainder_cheby print"<<std::endl;
  remainder_value.print();
  */
  return quotient_value;
  
}
  
//ChebyshevPoly hecate::ChebyshevPoly::divide_remainder(const ChebyshevPoly& denominator_cheby) {
ChebyshevPoly hecate::ChebyshevPoly::operator%(const ChebyshevPoly& denominator_cheby) {
  /* wrong divide!!! must change for chebyshev ******/
  std::vector<double> numerator = chebyshev_to_monomial(coefficients);
  std::vector<double> denominator = chebyshev_to_monomial(denominator_cheby.coefficients);

  std::vector<double> quotient;
  std::vector<double> remainder;
  poly_divide(numerator, denominator, quotient, remainder);
  
  // Convert quotient back to Chebyshev
  ChebyshevPoly remainder_value = ChebyshevPoly(monomial_to_chebyshev(remainder, remainder.size()-1));

  return remainder_value;
}

int hecate::ChebyshevPoly::coeff_size() {
    return coefficients.size();
}
double hecate::ChebyshevPoly::nth_coeff(int n) {
    return coefficients[n];
}

void hecate::ChebyshevPoly::print() const {
    for (const auto& c : coefficients) {
      std::cout<<c<<" ";
    }
    std::cout<<std::endl;
}
