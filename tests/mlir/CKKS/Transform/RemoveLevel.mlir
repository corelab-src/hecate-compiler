// RUN: hecate-opt %s -remove-level --ckks-config=%hecate_root/%ckks_config | FileCheck %s

module {
  func.func @test_remove_level(%arg0: tensor<1x!ckks.poly<2 * 16>>) -> tensor<1x!ckks.poly<2 * 4>> attributes {arg_scale = array<i64: 40>, btp_target = array<i64>, init_level = 16 : i64, res_scale = array<i64: 69>, selected_set = 1 : i64, smu0 = 0 : i64, smu_attached = true} {
    %0 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %1 = "ckks.modswitchc"(%0, %arg0) <{downFactor = 12 : i64}> : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 16>>) -> tensor<1x!ckks.poly<2 * 4>>
    %2 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %3 = "ckks.rotatec"(%2, %1) <{offset = array<i64: 0>}> : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    %4 = tensor.empty() : tensor<1x!ckks.poly<1 * 4>>
    %5 = "ckks.encode"(%4) <{level = 4 : i64, scale = 40 : i64, value = 0 : i64}> : (tensor<1x!ckks.poly<1 * 4>>) -> tensor<1x!ckks.poly<1 * 4>>
    %6 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %7 = "ckks.mulcp"(%6, %3, %5) : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<1 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    %8 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %9 = "ckks.rotatec"(%8, %1) <{offset = array<i64: 1>}> : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    %10 = tensor.empty() : tensor<1x!ckks.poly<1 * 4>>
    %11 = "ckks.encode"(%10) <{level = 4 : i64, scale = 40 : i64, value = 1 : i64}> : (tensor<1x!ckks.poly<1 * 4>>) -> tensor<1x!ckks.poly<1 * 4>>
    %12 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %13 = "ckks.mulcp"(%12, %9, %11) : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<1 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    %14 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %15 = "ckks.addcc"(%14, %7, %13) : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    %16 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %17 = "ckks.rotatec"(%16, %1) <{offset = array<i64: 2>}> : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    %18 = tensor.empty() : tensor<1x!ckks.poly<1 * 4>>
    %19 = "ckks.encode"(%18) <{level = 4 : i64, scale = 40 : i64, value = 2 : i64}> : (tensor<1x!ckks.poly<1 * 4>>) -> tensor<1x!ckks.poly<1 * 4>>
    %20 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %21 = "ckks.mulcp"(%20, %17, %19) : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<1 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    %22 = tensor.empty() : tensor<1x!ckks.poly<2 * 4>>
    %23 = "ckks.addcc"(%22, %15, %21) : (tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>, tensor<1x!ckks.poly<2 * 4>>) -> tensor<1x!ckks.poly<2 * 4>>
    return %23 : tensor<1x!ckks.poly<2 * 4>>
  }
}
// CHECK-LABEL: func.func @test_remove_level
// CHECK: "ckks.modswitchc"
// CHECK-NOT: ckks.poly<2 * 4>
// CHECK: "ckks.rotatec"
// CHECK: "ckks.encode"
// CHECK: "ckks.mulcp"
// CHECK: "ckks.addcc"
// CHECK: tensor<1x!ckks.poly<2 * 0>>
// CHECK-NOT: ckks.poly<2 * [^0]>
