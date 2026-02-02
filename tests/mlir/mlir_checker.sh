scriptPATH=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )

#### USE Method ############################
# PASS=estimate-latency ./mlir_checker.mlir
############################################

PASS="${PASS:-convert-earth-to-ckks}"
CONFIG="${CONFIG:-HEONGPU}"
ROOT="${ROOT:-$scriptPATH}"

name="${PASS}.mlir"

mapfile -t matches < <(
  find "$ROOT" -type f -name "$name" -print
)

if (( ${#matches[@]} == 0 )); then
  echo "[ERROR] No .mlir file found for PASS='$PASS' under '$ROOT'"
  echo "        Tried: $name"
  exit 1
elif (( ${#matches[@]} > 1 )); then
  echo "[ERROR] Multiple .mlir files match PASS='$PASS' (ambiguous):"
  printf "  - %s\n" "${matches[@]}"
  echo "Hint: set TARGET explicitly, or narrow ROOT/GROUP."
  exit 1
fi


TARGET="${TARGET:-${matches[0]}}"
../../build/bin/hecate-opt ${TARGET} \
    --allow-unregistered-dialect \
    -${PASS} \
    -split-input-file \
    --ckks-config=../../profiled_${CONFIG}_GPU.json

#../../build/bin/hecate-opt ./Earth/Conversion/convert_earth_to_ckks.mlir \
#    --allow-unregistered-dialect \
#    -convert-earth-to-ckks \
#    -split-input-file \
#    --ckks-config=../../profiled_HEONGPU_GPU.json
cd ${scriptPATH}
