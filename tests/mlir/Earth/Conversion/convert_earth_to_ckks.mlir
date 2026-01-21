// RUN: hecate-opt %s -convert-earth-to-ckks --ckks-config=%hecate_root/%ckks_config | FileCheck %s
  
module {
  func.func @test_func(%arg0: tensor<1x!earth.ci<47 * 11>>, %arg1: tensor<1x!earth.ci<47 * 11>>) -> tensor<1x!earth.ci<47 * 11>> {
    %0 = "earth.add"(%arg0, %arg1) : (tensor<1x!earth.ci<47 * 11>>, tensor<1x!earth.ci<47 * 11>>) -> tensor<1x!earth.ci<47 * 11>>
    return %0 : tensor<1x!earth.ci<47 * 11>>
  }
}

// CHECK-LABEL: func.func @test_func(
// CHECK-SAME:   %[[ARG0:.*]]: tensor<1x!ckks.poly<2 * 16>>,
// CHECK-SAME:   %[[ARG1:.*]]: tensor<1x!ckks.poly<2 * 16>>
// CHECK-SAME: ) -> tensor<1x!ckks.poly<2 * 16>> {
// CHECK:   %[[DST:.*]] = tensor.empty() : tensor<1x!ckks.poly<2 * 16>>
// CHECK:   %[[RESULT:.*]] = "ckks.addcc"(%[[DST]], %[[ARG0]], %[[ARG1]]) : (tensor<1x!ckks.poly<2 * 16>>, tensor<1x!ckks.poly<2 * 16>>, tensor<1x!ckks.poly<2 * 16>>) -> tensor<1x!ckks.poly<2 * 16>>
// CHECK:   return %[[RESULT]] : tensor<1x!ckks.poly<2 * 16>>
// CHECK: }
