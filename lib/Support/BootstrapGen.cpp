#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace hecate {

// Define shorthand for complex numbers
using Complex = std::complex<double>;
using Diagonal = std::vector<Complex>;
using DiagonalMatrix = std::vector<Diagonal>;

// Function to perform roll operation similar to numpy.roll
void roll_vector(std::vector<Complex> &vec, int shift) {
  int n = vec.size();
  if (n == 0)
    return;
  shift = ((shift % n) + n) % n; // Normalize shift
  std::rotate(vec.begin(), vec.begin() + n - shift, vec.end());
}

// Function to compute Ws(k)
std::vector<Complex> Ws(int k) {
  std::vector<Complex> result(k);
  for (int i = 0; i < k; ++i) {
    result[i] = std::exp(Complex(0, -M_PI / k * i));
  }
  return result;
}

// Function to repeat and arrange arrays similar to einops.repeat
std::vector<Complex> repeat_and_arrange(const std::vector<Complex> &arr1,
                                        const std::vector<Complex> &arr2,
                                        int repeats) {
  std::vector<Complex> result;
  for (int r = 0; r < repeats; ++r) {
    result.insert(result.end(), arr1.begin(), arr1.end());
    result.insert(result.end(), arr2.begin(), arr2.end());
  }
  return result;
}

// Function to perform cvMult (Diagonal Matrix-Vector multiplication)
std::vector<Complex> dvMult(const std::vector<Complex> &X, const StepMatrix &WS,
                            int bs) {
  // Diagonal MatVec multiplication
  int step = WS.second;
  DiagonalMatrix W = WS.first;

  int n = X.size();
  int W_rows = W.size();

  if (bs >= (W_rows + 1) / 2) {
    bs = (W_rows + 1) / 2;
  }
  int gs = (W_rows + 1) / bs;

  // Prepend zero vector to W
  DiagonalMatrix W_ext;
  W_ext.push_back(Diagonal(n, 0.0)); // Zero vector of size n
  W_ext.insert(W_ext.end(), W.begin(), W.end());

  int gs_half = gs / 2;

  // Roll W_reshaped
  for (int i = 0; i < gs; ++i) {
    int shift = step * bs * (i - gs_half);
    for (int j = 0; j < bs; ++j) {
      roll_vector(W_ext[i * bs + j], shift);
    }
  }

  // Generate Rotated vectors
  std::vector<std::vector<Complex>> Rotated(bs, X);
  for (int i = 0; i < bs; ++i) {
    int shift = -step * i;
    roll_vector(Rotated[i], shift);
  }

  // Compute subSum
  std::vector<std::vector<Complex>> subSum(gs, std::vector<Complex>(n, 0.0));
  for (int i = 0; i < gs; ++i) {
    for (int k = 0; k < n; ++k) {
      Complex sum = 0.0;
      for (int j = 0; j < bs; ++j) {
        sum += W_ext[i * bs + j][k] * Rotated[j][k];
      }
      subSum[i][k] = sum;
    }
  }

  // Roll subSum
  for (int i = 0; i < gs; ++i) {
    int shift = -step * bs * (i - gs_half);
    roll_vector(subSum[i], shift);
  }

  // Sum over subSum to get the result
  std::vector<Complex> result(n, 0.0);
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < gs; ++i) {
      result[k] += subSum[i][k];
    }
  }

  return result;
}

// Function to perform cvMult (Diagonal Matrix-Vector multiplication)
mlir::Value dvMultGen(mlir::OpBuilder builder, mlir::Value X,
                      const StepMatrix &WS) {
  assert(WS.first.size() >= 3); // W must have at least 3 diagonal
  // Diagonal MatVec multiplication
  int step = WS.second;
  DiagonalMatrix W = WS.first;

  int n = hecate::earth::EarthDialect::polynomialDegree;
  int W_rows = W.size();

  int bs = std::exp2(int(std::log2(W_rows + 1)) / 2);
  int gs = (W_rows + 1) / bs;

  // Prepend zero vector to W
  DiagonalMatrix W_ext;
  W_ext.push_back(Diagonal(n, 0.0)); // Zero vector of size n
  W_ext.insert(W_ext.end(), W.begin(), W.end());

  int gs_half = gs / 2;

  // Roll W_reshaped
  for (int i = 0; i < gs; ++i) {
    int shift = step * bs * (i - gs_half);
    for (int j = 0; j < bs; ++j) {
      roll_vector(W_ext[i * bs + j], shift);
    }
  }

  // Generate Rotated vectors
  std::vector<mlir::Value> Rotated(bs, X);
  for (int i = 0; i < bs; ++i) {
    int shift = -step * i;
    Rotated[i] = builder.create<hecate::earth::RotateOp>(X.getLoc(), X, -shift);
  }

  // Compute subSum
  std::vector<mlir::Value> subSum(gs, X);
  for (int i = 0; i < gs; ++i) {
    subSum[i] = Rotated[0];
    for (int j = 1; j < bs; ++j) {
      auto WVal = builder.create<hecate::earth::ConstantOp>(
          X.getLoc(),
          llvm::ArrayRef(W_ext[i * bs + j].data(), W_ext[i * bs + j].size()));
      auto Mul =
          builder.create<hecate::earth::MulOp>(X.getLoc(), Rotated[j], WVal);
      subSum[i] =
          builder.create<hecate::earth::AddOp>(X.getLoc(), subSum[i], Mul);
    }
  }

  // Roll subSum
  for (int i = 0; i < gs; ++i) {
    int shift = -step * bs * (i - gs_half);
    subSum[i] =
        builder.create<hecate::earth::RotateOp>(X.getLoc(), subSum[i], -shift);
  }

  // Sum over subSum to get the result
  // std::vector<Complex> result(n, 0.0);
  mlir::Value result;
  for (int i = 0; i < gs; ++i) {
    if (i == 0) {
      result = subSum[i];
    } else {
      result =
          builder.create<hecate::earth::AddOp>(X.getLoc(), result, subSum[i]);
    }
  }

  return result;
}

// Function to perform ddMult (Diagonal Matrix-Matrix multiplication)
StepMatrix ddMult(const StepMatrix &WS0, const StepMatrix &WS1) {
  assert(WS1.first.size() == 3); // W1 must have 3 diagonal
  assert((WS0.first.size() + 1) / 2 == WS1.second);

  DiagonalMatrix W0 = WS0.first;
  int step = WS0.second;
  DiagonalMatrix W1 = WS1.first;

  int prev = W0.size();
  int num = W0.size() + 1;
  int n = W0[0].size();

  DiagonalMatrix result(prev + num, Diagonal(n, 0.0));
  for (int k = 0; k < 3; k++) {
    for (int i = 0; i < prev; ++i) {
      int shift = -step * ((int)i - (int)(prev / 2));
      Diagonal W1_k = W1[k];
      roll_vector(W1_k, shift);
      for (int j = 0; j < n; ++j) {
        result[(num / 2) * k + i][j] += W0[i][j] * W1_k[j];
      }
    }
  }
  return {result, step};
}

// Function to compute the inverse of a diagonal matrix
StepMatrix dInverse(const StepMatrix &WS, int n) {
  DiagonalMatrix W = WS.first;
  int step = WS.second;
  int num = W.size();
  DiagonalMatrix result(num, Diagonal(W[0].size()));
  for (int i = 0; i < num; ++i) {
    int shift = (i - num / 2) * step;
    result[num - i - 1] = W[i];
    roll_vector(result[num - i - 1], shift);
    // Conjugate
    for (auto &val : result[num - i - 1]) {
      val = std::conj(val) / double(n);
    }
  }
  return {result, step};
}

std::vector<StepMatrix> generateConstant(int n) {

  std::vector<int> k_values;
  for (int i = 0; i < std::log2(n); ++i) {
    k_values.push_back(static_cast<int>(std::pow(2, i)));
  }
  std::reverse(k_values.begin(), k_values.end()); // Reverse the vector

  // Compute ld, dd, ud, and dM
  std::vector<StepMatrix> dM(k_values.size());

  for (int i = 0; i < k_values.size(); i++) {
    int k = k_values[i];
    int repeats = n / 2 / k;
    Diagonal zeros_k(k, 0.0);
    Diagonal ones_k(k, 1.0);
    Diagonal Ws_k = Ws(k);
    Diagonal negative_Ws_k = Ws_k;
    for (auto &val : negative_Ws_k)
      val = -val;

    // ld
    dM[i].first.push_back(repeat_and_arrange(zeros_k, Ws_k, repeats));

    // dd
    dM[i].first.push_back(repeat_and_arrange(ones_k, negative_Ws_k, repeats));

    // ud
    dM[i].first.push_back(repeat_and_arrange(ones_k, zeros_k, repeats));
    dM[i].second = k;
  }

  return dM;
}

std::vector<StepMatrix> mergeMatrix(const std::vector<StepMatrix> &dM,
                                    std::vector<int> interval) {
  std::vector<StepMatrix> result;
  int start = 0;
  for (auto len : interval) {
    int end = start + len;
    StepMatrix merged = dM[end - 1];
    for (int j = end - 2; j >= start; j--) {
      merged = ddMult(merged, dM[j]);
    }
    result.push_back(merged);
    start = end;
  }
  return result;
}
// Example
//  int main() {
//    int n = 8; // Define n (must be a power of 2)
//    std::vector<int> k_values;
//    for (int i = 0; i < std::log2(n); ++i) {
//      k_values.push_back(static_cast<int>(std::pow(2, i)));
//    }
//    std::reverse(k_values.begin(), k_values.end()); // Reverse the vector
//
//    // Compute ld, dd, ud, and dM
//    std::vector<std::vector<Complex>> ld_list, dd_list, ud_list;
//    for (auto k : k_values) {
//      int repeats = n / 2 / k;
//      std::vector<Complex> zeros_k(k, 0.0);
//      std::vector<Complex> ones_k(k, 1.0);
//      std::vector<Complex> Ws_k = Ws(k);
//
//      // ld
//      std::vector<Complex> ld_vector = repeat_and_arrange(zeros_k, Ws_k,
//      repeats); ld_list.push_back(ld_vector);
//
//      // dd
//      std::vector<Complex> negative_Ws_k = Ws_k;
//      for (auto &val : negative_Ws_k)
//        val = -val;
//      std::vector<Complex> dd_vector =
//          repeat_and_arrange(ones_k, negative_Ws_k, repeats);
//      dd_list.push_back(dd_vector);
//
//      // ud
//      std::vector<Complex> ud_vector =
//          repeat_and_arrange(ones_k, zeros_k, repeats);
//      ud_list.push_back(ud_vector);
//    }
//
//    // Rearrange dM
//    int num_k = ld_list.size();
//    std::vector<std::vector<std::vector<Complex>>> dM(num_k);
//    for (int i = 0; i < num_k; ++i) {
//      dM[i].resize(3);
//      dM[i][0] = ld_list[i];
//      dM[i][1] = dd_list[i];
//      dM[i][2] = ud_list[i];
//    }
//
//    // Input vector x
//    std::vector<Complex> x(n);
//    for (int i = 0; i < n; ++i) {
//      x[i] = Complex(double(i) / n, double(n - i - 1) / n);
//    }
//
//    std::cout << "Input x:" << std::endl;
//    for (const auto &val : x) {
//      std::cout << val << std::endl;
//    }
//
//    // Perform FFT using cvMult
//    std::vector<Complex> fft_result = x;
//    for (size_t i = 0; i < dM.size(); ++i) {
//      fft_result = dvMult(fft_result, dM[i], k_values[i]);
//    }
//
//    std::cout << "\nFFT result:" << std::endl;
//    for (const auto &val : fft_result) {
//      std::cout << val << std::endl;
//    }
//
//    return 0;
//  }
} // namespace hecate
