
#ifndef HECATE_CONVERSION_PASSES_H
#define HECATE_CONVERSION_PASSES_H

#include "hecate/Conversion/CKKSToCKKS/ArithToDpsArith.h"
#include "hecate/Conversion/CKKSToCKKS/SCFToDpsSCF.h"
#include "hecate/Conversion/CKKSToCKKS/UpscaleToMulcp.h"
#include "hecate/Conversion/EarthToCKKS/EarthToCKKS.h"
#include "hecate/Conversion/EarthToEarth/EarthToEarth.h"

namespace hecate {

#define GEN_PASS_REGISTRATION
#include "hecate/Conversion/Passes.h.inc"

} // namespace hecate

#endif
