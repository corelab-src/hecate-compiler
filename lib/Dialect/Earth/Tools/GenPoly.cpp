#include "hecate/Dialect/Earth/Tools/GenPoly.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Tools/Chebyshev.h"
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

using namespace hecate;

void print_PolyVector(
    std::vector<std::vector<std::variant<hecate::ChebyshevPoly, int>>>
        chebyshevPolys) {
  // print chebyshev vector for debug
  std::cout << "[";
  for (size_t i = 0; i < chebyshevPolys.size(); i++) {
    std::cout << "[";
    auto chebyshev_cousin = chebyshevPolys[i];
    for (size_t j = 0; j < chebyshev_cousin.size(); j++) {
      std::visit(
          [](auto &&arg) {
            if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, int>) {
              std::cout << arg << " ";
            } else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                                ChebyshevPoly>) {
              std::cout << "{";
              arg.print();
              std::cout << "}";
            }
          },
          chebyshev_cousin[j]);
    }
    std::cout << "]";
  }
  std::cout << "]" << std::endl;
}

hecate::GenPoly::GenPoly() {
  std::cout << "genPoly Basic" << std::endl;
  std::cout << "degree : " << length << std::endl;
  tree_var = LoadVar("treeStr.txt");
  coeff_var = LoadVar("coeffStr.txt");
  GenPoly_run();
}

hecate::GenPoly::GenPoly(int degree) {
  std::cout << "genPoly Test" << std::endl;
  std::cout << "degree : " << degree << std::endl;
  tree_var = LoadVar("treeStr.txt");
  coeff_var = LoadVar("coeffStr.txt");
  length = degree;
  GenPoly_run();
  // GenPoly(tree_var, coeff_var, degree);
}

hecate::GenPoly::GenPoly(const std::string &treeStr,
                         const std::string &coeffStr, int degree,
                         float scale_in) {
  std::cout << "genPoly fileload" << std::endl;
  std::cout << "degree : " << degree << std::endl;
  tree_var = LoadVar(treeStr);
  coeff_var = LoadVar(coeffStr);
  length = degree;
  scale = scale_in;
  GenPoly_run();
}

hecate::GenPoly::GenPoly(const std::vector<std::string> &treeVar,
                         const std::vector<std::string> &coeffVar, int degree,
                         float scale_in) {
  std::cout << "genPoly 4" << std::endl;
  std::cout << "degree : " << degree << std::endl;
  tree_var = treeVar;
  coeff_var = coeffVar;
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
  std::cout << "GenPoly Run" << std::endl;
  std::cout << "degree : " << length << ", scale : " << scale << std::endl;

  // new tree
  std::vector<std::vector<int>> new_tree;
  std::vector<int> line_tree;
  for (const auto &line : tree_var) {
    std::vector<int> line_tree;
    std::istringstream stream(line);
    int value;

    while (stream >> value) {
      line_tree.push_back(value);
    }
    new_tree.push_back(line_tree);
  }

  // coeff, cheby_mish
  // DK : Please check the double type. Does it affect the result? (ex.
  // -0.30004081084089734406e-28 -> -3.00041e-29)
  std::vector<double> coeff;
  for (const auto &line : coeff_var) {
    std::istringstream stream(line);
    double value;
    if (stream >> value) {
      coeff.push_back(value / scale);
    }
  }

  Calc_Chebyshev(new_tree, coeff);
  // auto calc_order = Calc_Chebyshev(new_tree, coeff);
  /*
   for (const auto& order : calc_order) {
         std::cout << "Calc Order: " << std::get<0>(order) << ", "
                   << std::get<1>(order) << ", " << std::get<2>(order) << ", "
                   << std::get<3>(order) << std::endl;
     }
 */
  // for (const auto& num : coeff) {
  //   std::cout << num << std::endl;
  // }
}

std::vector<std::string> hecate::GenPoly::LoadVar(const std::string &filename) {
  std::cout << "Load " << filename << "_var" << std::endl;

  // {ProjectRoot}/python/poly/poly/data/*.txt
  std::filesystem::path ProjectRoot =
      std::filesystem::absolute(std::filesystem::path(__FILE__)
                                    .parent_path()
                                    .parent_path()
                                    .parent_path()
                                    .parent_path()
                                    .parent_path());
  std::filesystem::path PolyData_path = std::filesystem::absolute(
      ProjectRoot / "python" / "poly" / "poly" / "data");
  std::cout << PolyData_path.string() << std::endl;
  std::string File_Path = (PolyData_path / filename).string();

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
    // exit();
  }

  // Check result
  // std::cout << filename <<" :" << std::endl;
  // for (const auto& line : file_var) {
  //  std::cout << line;
  //}
  return file_var;
}

void hecate::GenPoly::Calc_Chebyshev(std::vector<std::vector<int>> tree,
                                     std::vector<double> coeff) {
  std::cout << "Chebyshev" << std::endl;
  ChebyshevPoly cheby_mish(coeff);
  std::vector<std::vector<std::variant<ChebyshevPoly, int>>> chebyshevPolys;

  std::cout << "Chebyshev Poly" << std::endl;
  std::vector<std::variant<ChebyshevPoly, int>> chebyshev_cousin;

  // create same tree
  for (size_t i = 0; i < tree.size(); i++) {
    const auto &cousins = tree[i];
    std::vector<std::variant<ChebyshevPoly, int>> chebyshev_cousin;
    for (size_t j = 0; j < cousins.size(); j++) {
      chebyshev_cousin.push_back(cousins[j]);
    }
    chebyshevPolys.push_back(chebyshev_cousin);
  }

  // print_PolyVector(chebyshevPolys);
  chebyshevPolys[0][0] = cheby_mish;

  for (size_t i = 0; i < tree.size(); i++) {
    const auto &cousins = tree[i];
    for (size_t j = 0; j < cousins.size(); j++) {
      int divisor = cousins[j];
      if (divisor <= 0) {
        // std::cout<<"divisor is <= 0 :"<<tree[i][j]<<std::endl;
        continue;
      } else {
        std::vector<double> divisor_coeffs(divisor, 0.0);
        divisor_coeffs.push_back(1.0);
        ChebyshevPoly divisor_poly(divisor_coeffs);

        std::visit(
            [&divisor_poly, &chebyshevPolys, i, j](auto &&arg) {
              if constexpr (std::is_same_v<std::decay_t<decltype(arg)>, int>) {
                std::cerr << "push_back error " << arg << std::endl;
              } else if constexpr (std::is_same_v<std::decay_t<decltype(arg)>,
                                                  ChebyshevPoly>) {
                // chebyshevPolys[i+1][2*j+1]=arg.divide_quotient(divisor_poly);
                // chebyshevPolys[i+1][2*j]=arg.divide_remainder(divisor_poly);
                chebyshevPolys[i + 1][2 * j + 1] = arg / divisor_poly;
                chebyshevPolys[i + 1][2 * j] = arg % divisor_poly;
              }
            },
            chebyshevPolys[i][j]);

        // print_PolyVector(chebyshevPolys);
      }
    }
  }

  std::cout << "calc_order" << std::endl;
  // std::vector<std::tuple<int, int, int, ChebyshevPoly>> calc_order;
  for (size_t i = 0; i < tree.size(); ++i) {
    for (size_t j = 0; j < tree[i].size(); ++j) {
      int divisor = tree[i][j];
      if (divisor >= 0) {
        calc_order.emplace_back(i, j, divisor,
                                std::get<ChebyshevPoly>(chebyshevPolys[i][j]));
      }
    }
  }

  // Reverse order
  std::sort(calc_order.begin(), calc_order.end(),
            [](const auto &a, const auto &b) {
              return std::get<0>(a) > std::get<0>(b);
            });

  /*
  for (size_t i=0;i<calc_order.size();i++) {
    std::cout<<std::get<0>(calc_order[i])<<" "<<std::get<1>(calc_order[i])<<"
  "<<std::get<2>(calc_order[i])<<std::endl; std::get<3>(calc_order[i]).print();
  }
  */
}

template <typename T, typename U>
std::vector<T> vector_add(const std::vector<T> &input, const U &rhs) {
  std::vector<T> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = input[i] + rhs;
  }
  return result;
}

template <typename T>
std::vector<T> vector_add(const std::vector<T> &input,
                          const std::vector<T> &rhs) {
  if (input.size() != rhs.size()) {
    std::cerr << "Vectors must be of the same size." << std::endl;
  }
  std::vector<T> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = input[i] + rhs[i];
  }
  return result;
}

template <typename T, typename U>
std::vector<T> vector_sub(const std::vector<T> &input, const U &rhs) {
  std::vector<T> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = input[i] - rhs;
  }
  return result;
}

template <typename T>
std::vector<T> vector_sub(const std::vector<T> &input,
                          const std::vector<T> &rhs) {
  if (input.size() != rhs.size()) {
    std::cerr << "Vectors must be of the same size." << std::endl;
  }
  std::vector<T> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = input[i] - rhs[i];
  }
  return result;
}

template <typename T, typename U>
std::vector<T> vector_mult(const std::vector<T> &input, const U &rhs) {
  std::vector<T> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = input[i] * rhs;
  }
  return result;
}

template <typename T>
std::vector<T> vector_mult(const std::vector<T> &input,
                           const std::vector<T> &rhs) {
  if (input.size() != rhs.size()) {
    std::cerr << "Vectors must be of the same size." << std::endl;
  }
  std::vector<T> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = input[i] * rhs[i];
  }
  return result;
}

void vector_print(std::vector<double> input) {
  // for (size_t i=0; i<10; i++) {
  for (size_t i = 0; i < input.size(); i++) {
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;
}

// For checking the values from GSBS
std::vector<double> hecate::GenPoly::GSBS_check(std::vector<double> input) {
  // int64_t hecate::GenPoly::GSBS_check() {
  std::cout << "Giant-Step, Baby-Step Test" << std::endl;
  std::vector<std::vector<double>> babyTs;
  // Temporarily, set input as a vector.
  std::map<int, std::vector<double>> giantTs;
  giantTs.insert(std::make_pair(0, 1)); // 0 is not use...
  giantTs.insert(std::make_pair(1, input));
  int count = int(floor(log2(length)));
  for (int i = 1; i < count; i++) {
    int idx = pow(2, i);
    int pre_idx = pow(2, i - 1);
    // giantTs[idx] = 2 * giantTs[pre_idx]  * giantTs[pre_idx] + -1
    auto value = vector_mult(giantTs[pre_idx], 2);
    value = vector_mult(value, giantTs[pre_idx]);
    value = vector_add(value, -1);
    giantTs.insert(std::make_pair(idx, value));
  }

  babyTs.push_back(input);
  for (int i = 1; i < count; i++) {
    int idx = pow(2, i);
    std::vector<std::vector<double>> babyAdd;
    for (size_t j = 0; j < babyTs.size(); j++) {
      // 2 * poly * giantTs[idx]
      auto poly = babyTs[j];
      auto value = vector_mult(poly, 2);
      value = vector_mult(value, giantTs[idx]);
      value = vector_sub(value, babyTs[babyTs.size() - 1 - j]); // new-old
      babyAdd.push_back(value);
    }
    babyTs.reserve(babyTs.size() + babyAdd.size());
    babyTs.insert(std::end(babyTs), std::begin(babyAdd), std::end(babyAdd));
    // currently not use?
    // std::vector<std::vector<int>> sdfs;
  }

  /*
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
  */
  // calc_order : i, j, deg, leaf
  std::map<std::tuple<int, int>, std::vector<double>> tmpPoly;
  for (size_t iter = 0; iter < calc_order.size(); iter++) {
    auto calc = calc_order[iter];
    int i = std::get<0>(calc);
    int j = std::get<1>(calc);
    int deg = std::get<2>(calc);

    if (std::get<2>(calc) == 0) {
      std::vector<double> poly(babyTs[0].size(),
                               0.0); // babyTs items have same size
      for (int k = 0; k < length / 2; k++) {
        if (std::get<3>(calc).coeff_size() > 2 * k + 1) {
          // poly += leaf.coef[2*k+1] * babyTs[k];
          auto value =
              vector_mult(babyTs[k], std::get<3>(calc).nth_coeff(2 * k + 1));
          poly = vector_add(poly, value);
        }
      }
      tmpPoly[std::make_tuple(i, j)] = poly;
    } else {
      if (giantTs.find(deg) == giantTs.end()) {
        // giantTs[deg] = 2* giantTs[deg//2] * giantTs[deg//2] + -1
        auto value = vector_mult(giantTs[deg / 2], 2);
        value = vector_mult(value, giantTs[deg / 2]);
        value = vector_sub(value, 1);
        giantTs.insert(std::make_pair(deg, value));
      }
      // tmpPoly[(i,j)] = tmpPoly[(i+1, 2*j+1)] * giantTs[deg] + tmpPoly[(i+1,
      // 2*j)]
      std::tuple<int, int> key = std::make_tuple(i, j);
      auto value = vector_mult(tmpPoly.at(std::make_tuple(i + 1, 2 * j + 1)),
                               giantTs[deg]);
      value = vector_add(value, tmpPoly.at(std::make_tuple(i + 1, 2 * j)));
      tmpPoly[std::make_tuple(i, j)] = value;
    }
    /*
    // print tmpPoly
    for (const auto& pair : tmpPoly) {
      std::cout<<"key: ("
    <<std::get<0>(pair.first)<<","<<std::get<1>(pair.first)<<"):";
      vector_print(pair.second);
    }
    */
  }
  return tmpPoly.at(std::make_tuple(0, 0));
}

// For creating HE operations for compiler
mlir::Value hecate::GenPoly::GSBS_createHEOps(mlir::OpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value input) {
  std::cout << "Giant-Step, Baby-Step Test" << std::endl;

  // std::vector<hecate::earth::CipherType> babyTs;
  // std::map<int, hecate::earth::CipherType> giantTs;
  std::vector<mlir::Value> babyTs;
  std::map<int, mlir::Value> giantTs;

  giantTs[1] = input;

  int count = int(floor(log2(length)));
  // auto one = builder.create<hecate::earth::ConstantOp>(loc,
  // llvm::ArrayRef<double>({1.0}));
  // one.setValueAttr(builder.getI64IntegerAttr(11)); // Temporarily without
  // ElideConstant. Need to Fix!!!
  auto two = builder.create<hecate::earth::ConstantOp>(
      loc, llvm::ArrayRef<double>({2.0}));
  two.setValueAttr(builder.getI64IntegerAttr(
      22)); // Temporarily without ElideConstant. Need to Fix!!!
  auto minus_one = builder.create<hecate::earth::ConstantOp>(
      loc, llvm::ArrayRef<double>({-1.0}));
  minus_one.setValueAttr(builder.getI64IntegerAttr(
      33)); // Temporarily without ElideConstant. Need to Fix!!!
  for (int i = 1; i < count; i++) {
    int idx = pow(2, i);
    int pre_idx = pow(2, i - 1);
    std::cout << idx << " " << pre_idx << std::endl;
    // giantTs[idx] = 2 * giantTs[pre_idx]  * giantTs[pre_idx] + -1
    mlir::Value preValue = giantTs[pre_idx];
    mlir::Value two_preValue =
        builder.create<hecate::earth::MulOp>(loc, preValue, two);
    mlir::Value two_ppreValue =
        builder.create<hecate::earth::MulOp>(loc, two_preValue, preValue);
    giantTs[idx] =
        builder.create<hecate::earth::AddOp>(loc, two_ppreValue, minus_one);
  }

  // babyTs
  babyTs.push_back(input);
  for (int i = 1; i < count; i++) {
    int idx = pow(2, i);
    std::vector<mlir::Value> babyAdd;
    for (size_t j = 0; j < babyTs.size(); j++) {
      // 2 * poly * giantTs[idx]
      mlir::Value poly = babyTs[j];
      mlir::Value two_poly =
          builder.create<hecate::earth::MulOp>(loc, poly, two);
      mlir::Value two_poly_giantTs =
          builder.create<hecate::earth::MulOp>(loc, two_poly, giantTs[idx]);

      mlir::Value subValue = babyTs[babyTs.size() - 1 - j];
      mlir::Value negateValue =
          builder.create<hecate::earth::NegateOp>(loc, subValue);
      mlir::Value new_minus_old = builder.create<hecate::earth::AddOp>(
          loc, two_poly_giantTs, negateValue);
      babyAdd.push_back(new_minus_old);
    }
    babyTs.insert(babyTs.end(), babyAdd.begin(), babyAdd.end());
  }

  // tmpPoly
  //  calc_order : i, j, deg, leaf
  std::map<std::tuple<int, int>, mlir::Value> tmpPoly;
  for (size_t iter = 0; iter < calc_order.size(); iter++) {
    auto calc = calc_order[iter];
    int i = std::get<0>(calc);
    int j = std::get<1>(calc);
    int deg = std::get<2>(calc);
    auto leaf = std::get<3>(calc);

    if (deg == 0) {
      mlir::Value poly = nullptr;
      for (int k = 0; k < length / 2; k++) {
        if (leaf.coeff_size() > 2 * k + 1) {
          // poly+= leaf.coef[2*k+1] * babyTs[k]
          double coeff = leaf.nth_coeff(2 * k + 1);
          auto coeffValue = builder.create<hecate::earth::ConstantOp>(
              loc, llvm::ArrayRef<double>(&coeff, 1));
          coeffValue.setValueAttr(builder.getI64IntegerAttr(
              44)); // Temporarily without ElideConstant. Need to Fix!!!
          mlir::Value babyValue = babyTs[k];
          mlir::Value coeff_babyValue =
              builder.create<hecate::earth::MulOp>(loc, babyValue, coeffValue);
          if (!poly) {
            poly = coeff_babyValue;
          } else {
            poly = builder.create<hecate::earth::AddOp>(loc, poly,
                                                        coeff_babyValue);
          }
        }
      }
      tmpPoly[std::make_tuple(i, j)] = poly;
    } else {
      if (giantTs.find(deg) == giantTs.end()) { // if not deg in giantTs
        // giantTs[deg] = 2* giantTs[deg//2] * giantTs[deg//2] + -1
        mlir::Value preValue = giantTs[deg / 2];
        mlir::Value two_preValue =
            builder.create<hecate::earth::MulOp>(loc, preValue, two);
        mlir::Value two_ppreValue =
            builder.create<hecate::earth::MulOp>(loc, two_preValue, preValue);
        giantTs[deg] =
            builder.create<hecate::earth::AddOp>(loc, two_ppreValue, minus_one);
      }
      // tmpPoly[(i,j)] = tmpPoly[(i+1, 2*j+1)] * giantTs[deg] + tmpPoly[(i+1,
      // 2*j)]
      mlir::Value value1 = tmpPoly[std::make_tuple(i + 1, 2 * j + 1)];
      mlir::Value value2 = tmpPoly[std::make_tuple(i + 1, 2 * j)];
      mlir::Value tmpPoly_giant =
          builder.create<hecate::earth::MulOp>(loc, value1, giantTs[deg]);
      mlir::Value result =
          builder.create<hecate::earth::AddOp>(loc, tmpPoly_giant, value2);
      tmpPoly[std::make_tuple(i, j)] = result;
    }
  }
  return tmpPoly[std::make_tuple(0, 0)];
}
