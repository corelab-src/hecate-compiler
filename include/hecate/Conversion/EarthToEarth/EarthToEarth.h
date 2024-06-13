
#ifndef HECATE_CONVERSION_EARTHTOEARTH_EARTHTOEARTH_H
#define HECATE_CONVERSION_EARTHTOEARTH_EARTHTOEARTH_H

#include <memory>

#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"

namespace mlir {
namespace func {
class FuncOp;
}
class RewritePatternSet;
template <typename T> class OperationPass;
} // namespace mlir

namespace hecate {

#define GEN_PASS_DECL_EARTHTOEARTHCONVERSION
#include "mlir/Conversion/Passes.h.inc"

namespace earth {
std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createEarthToEarthConversionPass();

void populateEarthToEarthConversionPatterns(mlir::MLIRContext *ctxt,
                                            mlir::RewritePatternSet &patterns);

} // namespace earth

} // namespace hecate

#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H
