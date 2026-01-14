# Hecate Compiler — Supported Operations & Patterns

This document lists the currently supported operations and fusion patterns in the Hecate compiler. It is organized by **Frontend (Python-like API)**, **Midend (MLIR dialect ops by role/strategy)**, **Fusion patterns**, and **Backend (Earth) notes**.

---

## Strategy & Layout Selection

- **Preferred attribute (op-local):**
  - `hecate.layout = { strategy = "multiplexed" }`

---

## Front-End (Python-like) Operations

The Hecate Python-facing API mirrors common DL ops. The table below summarizes status and notes for those wrappers (user-level view).

| Op          | Supported | WIP | Use | Notes | Opcode |
|-------------|-----------|-----|---|---|---|
| `Add`        | O        |     | `+` ||
| `Sub`        | O        |     | `-` ||
| `Mul`        | O        |     | `*` ||
| `Div`        | △        |     | `/` | Only constant divisor ||
| `Rotate`     | X        |     | `x.rotate(offset)` | Slot rotation via Earth rotate  ||
| `SiLU`       | O        |     | `x.SiLU()` | | 24 |
| `BatchNorm1d`| O        |     | `x.BatchNorm1d(BNweight, num_features)` | Current, it is abstractBN without divide ||
| `BatchNorm2d`| O        |     | `x.BatchNorm2d(BNweight, num_features)` | Current, it is abstractBN without divide | 21 |
| `Conv2d`     | O        |     | `x.Conv2d(weight, in_channels, out_channels, kernel_size, stride, padding, groups, bias)`  | | 22 |
| `AdaptiveAvgPool2d`  | △ |  | `x.AdaptiveAvgPool2d(output_size)` | Currently, only output_size=1 | 29 |
| `AvgPool2d`  | O        |  | `x.AvgPool2d(kernel_size, stride, padding)` |  | 23 |
| `Downsample2d` | △      | WIP | `x.Downsample2d(index)`, `x[:,:,::i,::j]` | | 25 |
| `cat`     | △        |     | `x.cat(tensors, dim))`  | Currently, only dim=1 supported | 28 |
| `Linear`       | O        |     | `x.Linear(in_features, out_features, bias)` | | 27 |
| `Pad`        |        | WIP | `x.pad(padding)` | Currently, the op exists in fusion| 26 |
| `Matmul`        |        | WIP | `x.Linear(weight, bias_weight, in_feature, out_feature)` | Currently, the op exists in fusion ||

---

## Middle-End: Dialect Ops by Role / Strategy

This section covers the **MLIR ops** the pipeline recognizes/lowers, whether they are **Common** (strategy-agnostic), **Base** wrappers (strategy switch at pattern), or **Multiplexed** only (impl lives under the multiplexed namespace). “Anchor” indicates whether a pattern uses the op as the primary replacement point.

### Arith

| Dialect.Op            | Status | Strategy | Anchor | Notes |
|-----------------------|--------|----------------|--------|-------|
| `arith.constant`      | O     | Common          | `arith.constant` | Tensor constants to plain/cipher |


### Tensor

| Dialect.Op               | Status | Strategy | Anchor | Notes |
|--------------------------|--------|----------------|--------|-------|
| `tensor.pad`             |      |  | `tensor.pad` |  |
| `tensor.extract_slice`   |      |        | `tensor.extract_slice` |  |
| `tensor.empty`           |      |          |       |  |
| `tensor.concat`          |  △  |          | `tensor.concat` | Currently, only dim=1 supported |
| `tensor.collapse_shape`          |     |          |       |  |
| `tensor.expand_shape`          |     |          |       |  |
| `tensor.transpose`          |     |          |       |  |

### Linalg

| Dialect.Op                      | Status | Strategy | Anchor | Notes |
|---------------------------------|--------|------------------|--------|-------|
| `linalg.conv_2d_nchw_fchw`      | O | Multiplexed | `linalg.conv_2d_nchw_fchw` |  |
| `linalg.pooling_nchw_sum`       | O | Multiplexed | `linalg.pooling_nchw_sum` |  |
| `linalg.matmul` | △  | Multiplexed | | |
| `linalg.generic` | △ |          | `linalg.generic` | Generic Op |

### Dev

| Dialect.Op         | Status | Strategy | Notes |
|--------------------|--------|---------|-------|
| `hecate.dev.print` | O | Common  | Debug print on Earth.plain/cipher. |


### Fusion Patterns

**Temporary**
**After applying Fusion analysis, the rules will be changed.**

Fusion patterns in this pipeline follow two rules:

1. **Anchor = final consumer.**  
   Each fusion is defined around the last op in the chain (**anchor**).  
   The anchor pattern walks backward through its inputs and absorbs known producers
   (e.g. Conv, BiasAdd, PoolingSum) as long as they are single-use and structurally
   match the expected form.

2. **Pre-shaping folds into the producer; post-processing fuses into the consumer.**  
   Ops that only prepare inputs (e.g. `tensor.pad` before `linalg.conv_2d_nchw_fchw`)
   are canonicalized into the producer’s parameters, so they do not create separate
   fusion variants. Ops that post-process the result (Bias, Abstract BN, ...)
   are fused by the anchor and then replaced with a single fused pattern.

If fusion succeeds, the anchor op is replaced with one fused op and the absorbed
producers are erased. If fusion is not legal (fanout, unexpected shape, etc.), each
op is lowered individually by its fallback pattern.

| Fusion                                | Status | Strategy | Anchor Op                    | Notes |
|---------------------------------------|---|---|------------------------------|-------|
| **Conv2D**                           |  |  |  |  |
| Conv2D + ConvBias                | X | Multiplexed | `linalg.generic` (ConvBias)| (Pad)-Conv2d-ConvBias |
| Conv2D + Abstract BN              | O | Multiplexed | `linalg.generic` (BN-like)| (Pad)-Conv2d-AbstractBN |
| Conv2D + ConvBias + Abstract BN  | X | Multiplexed | `linalg.generic` (BN-like)| (Pad)-Conv2d-ConvBias-AbstractBN |
| **Downsample**                           |  |  |  |  |
| Downsample (slice) + Channel Pad  | O | Multiplexed | `tensor.pad` | Downsample-pad |
| **AvgPool2D**                           |  |  |  |  |
| AvgPool2D = PoolingSum + Divide   | O | Multiplexed | `linalg.generic` | pooling_nchw_sum - generic(Div) |
| **Linear**                           |  |  |  |  |
| Linear   | O | Multiplexed | `linalg.matmul` | (collapse_shape) - matmul - generic(Add)-(expand_shape) |

> Pattern priorities: fusion patterns register with higher benefit than leaf patterns to ensure fusion fires first when applicable.


