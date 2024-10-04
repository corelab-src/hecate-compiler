#include "hecate/Dialect/Earth/Analysis/PolynomialAnalysis.h"
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
#include <algorithm>
#include <variant>

using namespace hecate;

class ChebyshevPoly {
public:
  ChebyshevPoly(const std::vector<double>& coeff) : coefficients(coeff) {}

  double evaluate(double x) const {
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

  //std::vector<double> divide(const std::vector<double>& numerator, const std::vector<double>& denominator) {
  ChebyshevPoly divide_quotient(const ChebyshevPoly& denominator_cheby) {
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
  
  ChebyshevPoly divide_remainder(const ChebyshevPoly& denominator_cheby) {
    /* wrong divide!!! must change for chebyshev ******/
    /* simple skeleton for run */
    int denom_size = denominator_cheby.coefficients.size();
    std::vector<double> temp(denom_size-1, 0.0);
    temp.push_back(1.0);
    return ChebyshevPoly(temp);
  }

  void print() const {
    for (const auto& c : coefficients) {
      std::cout<<c<<" ";
    }
    std::cout<<std::endl;
  }

private:
  std::vector<double> coefficients;
};


void print_PolyVector(std::vector<std::vector<std::variant<ChebyshevPoly, int>>> chebyshevPolys) {
  // print chebyshev vector for debug
  std::cout<<"[";
  for (size_t i = 0; i<chebyshevPolys.size(); i++) {
    std::cout<<"[";
    auto chebyshev_cousin = chebyshevPolys[i];
    for (size_t j = 0; j<chebyshev_cousin.size(); j++) {
      std::visit([](auto&& arg) {
        if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, int>) {
          std::cout<<arg<<" ";
        }
        else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, ChebyshevPoly>) {
          std::cout<<"{";
          arg.print();
          std::cout<<"}";
        }
      }, chebyshev_cousin[j]);
    }
    std::cout<<"]";
  }
  std::cout<<"]"<<std::endl;
}



hecate::PolynomialAnalysis::PolynomialAnalysis(mlir::Operation *op)
  : _op(op) {
}

int64_t hecate::PolynomialAnalysis::GenPoly_Test(int degree) {
  std::cout<<"genPoly Test"<<std::endl;
  std::cout<<"degree : "<<degree<<std::endl;
  auto tree_var = LoadVar("treeStr.txt");
  auto coeff_var = LoadVar("coeffStr.txt");
  GenPoly(tree_var, coeff_var, degree);

  return 0;
}
int64_t hecate::PolynomialAnalysis::GenPoly(const std::vector<std::string> &tree_var,
                       const std::vector<std::string> &coeff_var,
                       int degree,
                       float scale) {
  std::cout<<"GenPoly"<<std::endl;
  std::cout<<"degree : "<<degree<<", scale : "<<scale<<std::endl;
  // new tree
  std::vector <std::vector<int>> new_tree;
  std::vector <int> line_tree;
  for (const auto& line : tree_var) {
    std::vector<int> line_tree;
    std::istringstream stream(line);
    int value;

    while (stream >> value) {
      line_tree.push_back(value);
    }
    new_tree.push_back(line_tree);
  }

  // coeff, cheby_mish
  // DK : Please check the double type. Does it affect the result? (ex. -0.30004081084089734406e-28 -> -3.00041e-29)
  std::vector<double> coeff;
  for (const auto& line : coeff_var) {
    std::istringstream stream(line);
    double value;
    if (stream >> value) {
      coeff.push_back(value / scale);
    }
  }

  auto calc_order = Chebyshev(new_tree, coeff);
 /* 
  for (const auto& order : calc_order) {
        std::cout << "Calc Order: " << std::get<0>(order) << ", " 
                  << std::get<1>(order) << ", " << std::get<2>(order) << ", "
                  << std::get<3>(order) << std::endl;
    }
*/
  //for (const auto& num : coeff) {
  //  std::cout << num << std::endl;
  //}


  return 0;
}


std::vector<std::string> hecate::PolynomialAnalysis::LoadVar(const std::string &filename) {
  std::cout<<"Load "<<filename<<"_var"<<std::endl;
 
  // {ProjectRoot}/python/poly/poly/data/*.txt
  std::filesystem::path ProjectRoot = std::filesystem::absolute(std::filesystem::path(__FILE__).parent_path().parent_path().parent_path().parent_path().parent_path());
  std::filesystem::path PolyData_path = std::filesystem::absolute(ProjectRoot/"python"/"poly"/"poly"/"data");
  std::cout<<PolyData_path.string()<<std::endl;
  std::string File_Path = (PolyData_path/filename).string();
  
  // read file
  std::vector<std::string> file_var;
  std::ifstream File(File_Path);
  if (File.is_open()) {
    std::string line;
    while (std::getline(File, line)) {
      file_var.push_back(line);
    }
      File.close();
  } else {
    std::cerr << "Unable to open file: " << File_Path << std::endl;
    //exit();
  }

  // Check result
  //std::cout << filename <<" :" << std::endl;
  //for (const auto& line : file_var) {
  //  std::cout << line;
  //}
  return file_var;
}

//std::vector<std::tuple<int, int, int, int>> hecate::PolynomialAnalysis::Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff) {
int64_t hecate::PolynomialAnalysis::Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff) {
  std::cout<<"Chebyshev"<<std::endl;
  ChebyshevPoly cheby_mish(coeff);
  std::vector<std::vector<std::variant<ChebyshevPoly, int>>> chebyshevPolys;
  
  std::cout<<"Chebyshev Poly"<<std::endl;
  std::vector<std::variant<ChebyshevPoly, int>> chebyshev_cousin;

  // create same tree
  for (size_t i = 0; i<tree.size(); i++) {
    const auto& cousins = tree[i];
    std::vector<std::variant<ChebyshevPoly, int>> chebyshev_cousin;
    for (size_t j = 0; j < cousins.size(); j++) {
      chebyshev_cousin.push_back(cousins[j]);
    }
    chebyshevPolys.push_back(chebyshev_cousin);
  }

  //print_PolyVector(chebyshevPolys);
  chebyshevPolys[0][0]= cheby_mish;
  

  for (size_t i = 0; i < tree.size(); i++) {
    const auto& cousins = tree[i];
    for (size_t j = 0; j < cousins.size(); j++) {
      int divisor = cousins[j];
      if(divisor <= 0) {
        //std::cout<<"divisor is <= 0 :"<<tree[i][j]<<std::endl;
        continue;
      }
      else {
        std::vector<double> divisor_coeffs(divisor, 0.0);
        divisor_coeffs.push_back(1.0);
        ChebyshevPoly divisor_poly(divisor_coeffs);
      
        std::visit([&divisor_poly, &chebyshevPolys, i, j](auto&& arg) {
          if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, int>) {
            std::cerr<<"push_back error "<<arg<<std::endl;
          }
          else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, ChebyshevPoly>) {
            chebyshevPolys[i+1][2*j+1]=arg.divide_quotient(divisor_poly);
            chebyshevPolys[i+1][2*j]=arg.divide_remainder(divisor_poly);
          }
        }, chebyshevPolys[i][j]);
        
        //print_PolyVector(chebyshevPolys);
      }
    }
  }
  
  std::cout<<"calc_order"<<std::endl;
  std::vector<std::tuple<int, int, int, ChebyshevPoly>> calc_order;
  for (size_t i = 0; i < tree.size(); ++i) {
    for (size_t j = 0; j < tree[i].size(); ++j) {
      int divisor = tree[i][j];
      if (divisor >= 0) {
        calc_order.emplace_back(i, j, divisor, std::get<ChebyshevPoly>(chebyshevPolys[i][j]));
      }
    }
  }

  for (size_t i=0;i<calc_order.size();i++) {
    std::cout<<std::get<0>(calc_order[i])<<" "<<std::get<1>(calc_order[i])<<" "<<std::get<2>(calc_order[i])<<std::endl;
    std::get<3>(calc_order[i]).print();
  }


  //return calc_order;
  return 0;
}

int64_t hecate::PolynomialAnalysis::GSBS() {
  std::cout<<"Giant-Step, Baby-Step Test"<<std::endl;
  return 0;
}
