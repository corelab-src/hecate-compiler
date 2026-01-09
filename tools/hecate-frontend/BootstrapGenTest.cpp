#include "hecate/Support/Support.h"
#include <iostream>

using namespace hecate;

int main() {
  int n = 4096; // Define n (must be a power of 2)
  auto dM = generateConstant(n);

  std::vector<Complex> x(n);
  for (int i = 0; i < n; ++i) {
    x[i] = Complex(double(i) / n, double(n - i - 1) / n);
  }

  // std::cout << "Input x:" << std::endl;
  // for (const auto &val : x) {
  //   std::cout << val << std::endl;
  // }

  // Perform FFT using cvMult
  std::vector<Complex> fft_result = x;
  auto DM = mergeMatrix(dM, {4, 4, 4});
  for (size_t i = 0; i < DM.size(); ++i) {
    fft_result = dvMult(fft_result, DM[i]);
  }
  std::vector<StepMatrix> IDM;
  DM = mergeMatrix(dM, {6, 6});
  for (int i = 0; i < DM.size(); ++i) {
    IDM.push_back(dInverse(DM[DM.size() - 1 - i], 64));
  }
  for (size_t i = 0; i < IDM.size(); ++i) {
    fft_result = dvMult(fft_result, IDM[i]);
  }

  // StepMatrix fft_merged = dM[dM.size() - 1];
  // for (int i = dM.size() - 2; i >= 0; i--) {
  //   fft_merged = ddMult(fft_merged, dM[i]);
  // }
  // fft_result = dvMult(x, fft_merged);
  // fft_result = dvMult(fft_result, dInverse(fft_merged, n));

  std::cout << "\nFFT result:" << std::endl;
  double rms = 0.0;
  for (int i = 0; i < fft_result.size(); i++) {
    auto diff = fft_result[i].real() - x[i].real();
    rms += diff * diff;
    diff = fft_result[i].imag() - x[i].imag();
    rms += diff * diff;
  }
  std::cout << std::sqrt(rms / n) << std::endl;

  // fft_result = dvMult(dvMult(x, ddMult(dM[1], dM[0], 2), 2), dM[2]);
  //
  // std::cout << "\nFFT result:" << std::endl;
  // for (const auto &val : fft_result) {
  //   std::cout << val << std::endl;
  // }

  return 0;
}
