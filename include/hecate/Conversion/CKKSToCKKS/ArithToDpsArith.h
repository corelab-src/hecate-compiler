
#ifndef HECATE_CONVERSION_CKKSTOCKKS_ARITHTODPSARITH_H
#define HECATE_CONVERSION_CKKSTOCKKS_ARITHTODPSARITH_H

#include <memory>

#include "hecate/Conversion/CKKSCommon/PolyTypeConverter.h"

namespace mlir {
namespace func {
class FuncOp;
}
class RewritePatternSet;
template <typename T> class OperationPass;
} // namespace mlir

namespace hecate {

#define GEN_PASS_DECL_ARITHTODPSARITHCONVERSION
#include "mlir/Conversion/Passes.h.inc"

namespace ckks {

std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createArithToDpsArithConversionPass();

void populateArithToDpsArithConversionPatterns(
    mlir::MLIRContext *ctxt, mlir::RewritePatternSet &patterns);

} // namespace ckks

} // namespace hecate

#endif // MLIR_CONVERSION_ARITHTOLLVM_ARITHTOLLVM_H
