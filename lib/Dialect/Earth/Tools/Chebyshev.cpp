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

double hecate::ChebyshevPoly::evaluate(double x) const {
    double Tn = 0.0;
    double Tn1 = 1.0;
    double Tn2 = 0.0;

    for (size_t n = 0; n < coefficients.size(); ++n) {
      Tn = Tn1 * x * 2 - Tn2;
      Tn2 = Tn1;
      Tn1 = Tn;
    }

    double result = 0.0;
    for (size_t n = 0; n < coefficients.size(); ++n) {
      result += coefficients[n] * Tn;
    }
    return result;
}

ChebyshevPoly hecate::ChebyshevPoly::divide_quotient(const ChebyshevPoly& denominator_cheby) {
    /* wrong divide!!! must change for chebyshev ******/
    //std::cout<<"numerator print"<<std::endl;
    //print();
    //std::cout<<"denominator_cheby print"<<denominator_cheby.coefficients.size()<<std::endl;
    //denominator_cheby.print();
    
    // quotient = coefficients // denominator;
    auto denominator = denominator_cheby.coefficients;
    std::vector<double> quotient;
    if (denominator.empty() || (denominator.size() == 1 && denominator[0] == 0.0)) {
        std::cerr<<"Division by zero polynomial"<<std::endl;
        //exit();
        //throw std::invalid_argument("Division by zero polynomial");
    }

    // Perform polynomial long division
    std::vector<double> remainder = coefficients;
    while (remainder.size() >= denominator.size()) {
        // leading coefficients
        double coeffs = remainder.back() / denominator.back();
        quotient.push_back(coeffs);

        // Subtract the current term multiplied by the divisor
        for (size_t i = 0; i < denominator.size(); ++i) {
            remainder[remainder.size()-1-i] -= coeffs * denominator[denominator.size()-1-i];
        }
        // Remove the last term of remainder (it's effectively 0 after subtraction)
        remainder.pop_back();
    }

    std::reverse(quotient.begin(), quotient.end());  // Reverse to get correct order
    //std::cout<<"divide print"<<std::endl;
    //ChebyshevPoly(quotient).print();
    return ChebyshevPoly(quotient);
}
  
ChebyshevPoly hecate::ChebyshevPoly::divide_remainder(const ChebyshevPoly& denominator_cheby) {
    /* wrong divide!!! must change for chebyshev ******/
    /* simple skeleton for run */
    int denom_size = denominator_cheby.coefficients.size();
    std::vector<double> temp(denom_size-1, 0.0);
    temp.push_back(1.0);
    return ChebyshevPoly(temp);
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
