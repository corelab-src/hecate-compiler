// RUN: hecate-opt %s -estimate-latency --ckks-config=%hecate_root/%ckks_config | FileCheck %s

module {
  func.func @test_latency_estimator(%arg0: tensor<1x!earth.ci<40 * 1>>) -> tensor<1x!earth.ci<80 * 12>> attributes {arg_scale = array<i64: 40>, btp_target = array<i64>, init_level = 16 : i64, res_scale = array<i64: 69>, selected_set = 1 : i64, smu0 = 0 : i64, smu_attached = true} {
    %0 = "earth.modswitch"(%arg0) <{downFactor = 11 : i64}> : (tensor<1x!earth.ci<40 * 1>>) -> tensor<1x!earth.ci<40 * 12>>
    %1 = "earth.rotate"(%0) <{offset = array<i64: 0>}> : (tensor<1x!earth.ci<40 * 12>>) -> tensor<1x!earth.ci<40 * 12>>
    %2 = "earth.constant"() <{rms_var = 0.05349482071600755 : f64, value = 1 : i64}> : () -> tensor<1x!earth.pl<40 * 12>>
    %3 = "earth.mul"(%1, %2) : (tensor<1x!earth.ci<40 * 12>>, tensor<1x!earth.pl<40 * 12>>) -> tensor<1x!earth.ci<80 * 12>>
    return %3 : tensor<1x!earth.ci<80 * 12>>
  }
}

// This test has latency from profiled_HEONGPU_GPU.json (default).
// If user changes the profiled configuration, the value will be different.
// CHECK-LABEL: func.func @test_latency_estimator(
// CHECK: est_latency = 2.200000e+02 : f64
// CHECK-NOT: error
