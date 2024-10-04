#include "hecate/Dialect/Earth/Tools/GenPoly.h"
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

void print_PolyVector(std::vector<std::vector<std::variant<hecate::ChebyshevPoly, int>>> chebyshevPolys) {
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



hecate::GenPoly::GenPoly() {
  std::cout<<"genPoly Basic"<<std::endl;
  std::cout<<"degree : "<<length<<std::endl;
  tree_var = LoadVar("treeStr.txt");
  coeff_var = LoadVar("coeffStr.txt");
  GenPoly_run();

}

hecate::GenPoly::GenPoly(int degree) {
  std::cout<<"genPoly Test"<<std::endl;
  std::cout<<"degree : "<<degree<<std::endl;
  tree_var = LoadVar("treeStr.txt");
  coeff_var = LoadVar("coeffStr.txt");
  length = degree;
  GenPoly_run();
  //GenPoly(tree_var, coeff_var, degree);
}

hecate::GenPoly::GenPoly(const std::vector<std::string> &treeStr,
                       const std::vector<std::string> &coeffStr,
                       int degree,
                       float scale_in) {
  std::cout<<"genPoly 4"<<std::endl;
  std::cout<<"degree : "<<degree<<std::endl;
  tree_var = treeStr;
  coeff_var = coeffStr;
  length = degree;
  scale = scale_in;
  GenPoly_run();
}

/*
hecate::GenPoly::GenPoly_run(const std::vector<std::string> &tree_var,
                       const std::vector<std::string> &coeff_var,
                       int degree,
                       float scale) {
*/
void hecate::GenPoly::GenPoly_run() {
  std::cout<<"GenPoly Run"<<std::endl;
  std::cout<<"degree : "<<length<<", scale : "<<scale<<std::endl;

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

  Calc_Chebyshev(new_tree, coeff);
  //auto calc_order = Calc_Chebyshev(new_tree, coeff);
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
}


std::vector<std::string> hecate::GenPoly::LoadVar(const std::string &filename) {
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

int64_t hecate::GenPoly::Calc_Chebyshev(std::vector<std::vector<int>> tree, std::vector<double> coeff) {
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
  //std::vector<std::tuple<int, int, int, ChebyshevPoly>> calc_order;
  for (size_t i = 0; i < tree.size(); ++i) {
    for (size_t j = 0; j < tree[i].size(); ++j) {
      int divisor = tree[i][j];
      if (divisor >= 0) {
        calc_order.emplace_back(i, j, divisor, std::get<ChebyshevPoly>(chebyshevPolys[i][j]));
      }
    }
  }
 
  /* HERE... Where is reverse order.........................*/

  for (size_t i=0;i<calc_order.size();i++) {
    std::cout<<std::get<0>(calc_order[i])<<" "<<std::get<1>(calc_order[i])<<" "<<std::get<2>(calc_order[i])<<std::endl;
    std::get<3>(calc_order[i]).print();
  }


  //return calc_order;
  return 0;
}

std::vector<double> vector_add(std::vector<double> input, int rhs) {
  for (size_t i=0; i<input.size(); i++) {
    input[i] = input[i] + rhs;
  }
  return input;
}
std::vector<double> vector_sub(std::vector<double> input, int rhs) {
  for (size_t i=0; i<input.size(); i++) {
    input[i] = input[i] - rhs;
  }
  return input;
}
std::vector<double> vector_sub(std::vector<double> input, std::vector<double> rhs) {
  for (size_t i=0; i<input.size(); i++) {
    input[i] = input[i] - rhs[i];
  }
  return input;
}
std::vector<double> vector_mult(std::vector<double> input, int multiply) {
  for (size_t i=0; i<input.size(); i++) {
    input[i] = input[i] * multiply;
  }
  return input;
}
std::vector<double> vector_mult(std::vector<double> input, std::vector<double> multiply) {
  for (size_t i=0; i<input.size(); i++) {
    input[i] = input[i] * multiply[i];
  }
  return input;
}

void vector_print(std::vector<double> input) {
  for (size_t i=0; i<input.size(); i++) {
    std::cout<<input[i]<<" ";
  }
  std::cout<<std::endl;
}


// For checking the values from GSBS
int64_t hecate::GenPoly::GSBS_check(std::vector<double> input) {
//int64_t hecate::GenPoly::GSBS_check() {
  std::cout<<"Giant-Step, Baby-Step Test"<<std::endl;
  std::vector<std::vector<double>> babyTs;
  // Temporarily, set input as a vector.
  std::map<int, std::vector<double>> giantTs;
  giantTs.insert(std::make_pair(0, 1)); 
  giantTs.insert(std::make_pair(1, input)); 
  int count = int(floor(log2(length)));
  for (int i = 1; i < count; i++) {
    int idx = pow(2, i);
    int pre_idx = pow(2, i-1);
    std::cout<<idx<<" "<<pre_idx<<std::endl;
    //giantTs[idx] = 2 * giantTs[pre_idx]  * giantTs[pre_idx] + -1
    auto value = vector_mult(giantTs[pre_idx], 2);
    value = vector_mult(value, giantTs[pre_idx]);
    value = vector_add(value, -1);
    giantTs.insert(std::make_pair(idx, value));
  }
  
  babyTs.push_back(input); 
  for (int i = 1; i < count; i++) {
    int idx = pow(2, i);
    std::vector<std::vector<double>> babyAdd;
    for (size_t i=0; i<babyTs.size();i++) {
      // 2 * poly * giantTs[idx]
      auto poly = babyTs[i];
      auto value = vector_mult(poly, 2);
      value = vector_mult(value, giantTs[idx]);
      value = vector_sub(value, babyTs[babyTs.size()-1-i]); // new-old
      babyAdd.push_back(value);
    }
    babyTs.reserve(babyTs.size()+babyAdd.size());
    babyTs.insert(std::end(babyTs), std::begin(babyAdd), std::end(babyAdd));
    // currently not use?
    //std::vector<std::vector<int>> sdfs;
  }


  // print giantTs
  std::cout<<"giantTs"<<std::endl;
  for(auto iter = giantTs.begin(); iter != giantTs.end(); iter++){
    //std::cout<<iter->first<<" "<<iter->second<<std::endl;
    std::cout<<iter->first<<" : ";
    vector_print(iter->second);
  }
  // print babyTs
  std::cout<<"babyTs"<<std::endl;
  for(size_t i = 0; i < babyTs.size(); i++){
    std::cout<<i<<" : ";
    vector_print(babyTs[i]);
  }
  
  std::map<int, std::vector<double>> tmpPoly;
  
  return 0;
}

// For creating HE operations for compiler
//int64_t hecate::GenPoly::GSBS_createHEOps(mlir::RankedTensorType input) {
int64_t hecate::GenPoly::GSBS_createHEOps() {
  std::cout<<"Giant-Step, Baby-Step Test"<<std::endl;
  std::vector<int> babyTs;
  std::map<int, mlir::RankedTensorType> giantTs;
  
  int count = int(floor(log2(length)));
  for (int i = 1; i < count; i++) {
    int idx = pow(2, i);
    int pre_idx = pow(2, i-1);
    std::cout<<idx<<" "<<pre_idx<<std::endl;
    //giantTs[idx] = 2 * giantTs[pre_idx]  * giantTs[pre_idx] + -1
  }
  return 0;
}
