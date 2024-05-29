#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export HECATE=$( cd -- "$( dirname -- "$BASH_SOURCE[0]" )" &> /dev/null && pwd )
source .venv/bin/activate
source config.sh

alias hopt=$HECATE/build/bin/hecate-opt
build-hopt

# bench_name=(HarrisCornerDetection MLP LinearRegression SobelFilter PolynomialRegression Multivariate ResNet AlexNet VGG16 MobileNet SqueezeNet)
# bench_name=(SobelFilter HarrisCornerDetection MLP LinearRegression PolynomialRegression Multivariate)
# bench_name=(ResNet SqueezeNet MobileNet AlexNet VGG16)
bench_name=(LinearRegression PolynomialRegression Multivariate LogisticRegression SVM)
# bench_name=(PCA Kmeans)

waterline=(40)
# waterline=($(seq 30 40))
# compile_opt=(eva)
compile_opt=(halo halo_nonflex)
library=(HEAAN)
backend=(GPU)
epochs=(10 20)
# params=(FVa)

conf_name="ASPLOS25"
timestamp=$(date +%Y_%m_%d_%H%M%S)
mkdir "$DIR/results/$conf_name/$timestamp"
# timestamp=""

function compile(){
 # hopt-timing-only $1 $2 $3 >> $DIR/results/PLDI_24/$filename
 $HECATE/build/bin/hecate-opt --$1 --ckks-config="$HECATE/config.json" --waterline=$2  $HECATE/examples/traced/$3.mlir --mlir-timing 2> $DIR/results/$conf_name/tmp --mlir-disable-threading -o $HECATE/examples/optimized/$1/$3.$2.mlir
 # $HECATE/build/bin/hecate-opt --$1 --ckks-config="$HECATE/config.json" --waterline=$2  $HECATE/examples/traced/$3.mlir --mlir-timing --mlir-disable-threading -o $HECATE/examples/optimized/$1/$3.$2.mlir
 # $HECATE/build/bin/hecate-opt --$1 --ckks-config="$HECATE/profiled_$5_$6.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir
 # $HECATE/build/bin/hecate-opt --$1 --ckks-config="$HECATE/profiled_$5_$6.json" --waterline=$2 $HECATE/examples/traced/$3.mlir -o $HECATE/examples/optimized/$1/$3.$2.mlir  >> "$DIR/results/$conf_name/${timestamp}/$filename"
 return 0
}
function runner(){
 hc-test $1 $2 $3 $5 $6 $7 >> $DIR/results/$conf_name/${timestamp}/$4
 # hc-test $1 $2 $3 >> $DIR/results/$conf_name/tmp
 return 0
}
for name in ${bench_name[@]}
do
 for hw in ${backend[@]}
 do
  # filename=${name}-${compile_opt}.txt
  hc-trace $name
  for wtr in ${waterline[@]}
  do
   for opt in ${compile_opt[@]}
   do
    for ep in ${epochs[@]}
    do
     filename=${name}-${opt}-${wtr}-${ep}.txt
     for (( i = 0; i < 1; i++ ))
     do
      echo "Start compilation of ${name} ( waterline: ${wtr}, opt: ${opt})"
      compile $opt $wtr $name $filename $library $hw
      if [ $? -ne 0 ]; then
       echo "compile failed"
       continue
      fi
      echo "Complete compilation of ${name}_${wtr}_${opt}_${library}_${hw}"
      wait
      cstfile=$DIR/examples/traced/_hecate_${name}.cst
      hevmfile=$DIR/examples/optimized/${compile_opt}/${name}.${waterline}._hecate_${name}.hevm
      CST_SIZE=$(stat --printf=%s $cstfile)
      HEVM_SIZE=$(stat --printf=%s $hevmfile)
      FILESIZE=$((CST_SIZE + HEVM_SIZE))
      echo "cst_size: $CST_SIZE" >> $DIR/results/$conf_name/${timestamp}/$filename
      echo "hevm_size: $HEVM_SIZE" >> $DIR/results/$conf_name/${timestamp}/$filename
      echo "total_file_size: $FILESIZE" >> $DIR/results/$conf_name/${timestamp}/$filename

      echo "Process ${name}_${wtr}_${opt}_${library}_${hw}_${ep}"
      runner $opt $wtr $name $filename $library $hw $ep
      wait
      if [ $? -ne 0 ]; then
       echo "process failed"
       continue
      fi
      cat "$DIR/results/$conf_name/tmp" >> "$DIR/results/$conf_name/${timestamp}/$filename"
      echo "Results are recorded."
      echo ""
     done
    done
   done
  done
 done
done

##SEAL
# bench_name=(SobelFilter HarrisCornerDetection MLP LinearRegression PolynomialRegression Multivariate)
# bench_name=(SobelFilter HarrisCornerDetection)
bench_name=()

waterline=(30)
# waterline=($(seq 20 40))
compile_opt=(eva)

library=(SEAL)
backend=(CPU)

for hw in ${backend[@]}
do
 for name in ${bench_name[@]}
 do
  # filename=${name}-${compile_opt}.txt
  filename=${name}-${library}-${hw}.txt
  hc-trace $name
  for wtr in ${waterline[@]}
  do
   for opt in ${compile_opt[@]}
   do
    echo "Start compilation of ${name} ( waterline : ${wtr}, opt : ${opt} )"
    compile $opt $wtr $name $filename $library $hw
    if [ $? -ne 0 ]; then
     echo "compile failed"
     continue
    fi
    echo "Complete compilation of ${name}_${wtr}_${opt}_${library}_${hw}"
    for (( i = 0; i < 2; i++ ))
    do
     echo "Process ${name}_${wtr}_${opt}_${library}_${hw}"
     runner $opt $wtr $name $filename $library $hw
     if [ $? -ne 0 ]; then
      echo "process failed"
      continue
     fi
     # cat "$DIR/results/$conf_name/tmp" >> "$DIR/results/$conf_name/${timestamp}/$filename"
     echo "Results are recorded."
     echo ""
    done
   done
  done
 done
done


echo $timestamp
