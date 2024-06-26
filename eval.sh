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
# bench_name=(LinearRegression PolynomialRegression Multivariate LogisticRegression Kmeans SVM)
bench_name=(LinearRegression_Loop PolynomialRegression_Loop Multivariate_Loop SVM_Loop Kmeans_Loop LogisticRegression_Loop)
# bench_name=(LinearRegression_Loop PolynomialRegression_Loop Multivariate_Loop)
# bench_name=(SVM_Loop Kmeans_Loop LogisticRegression_Loop)
# bench_name=(LogisticRegression_Loop)
# bench_name=(LinearRegression)
# bench_name=()

waterline=(40)
# waterline=($(seq 30 40))
# compile_opt=(eva)
compile_opt=(simple_loop packed_loop unroll_loop flex_loop packed_unroll_loop packed_flex_loop packed_unroll_flex_loop)
# compile_opt=(packed_unroll_flex_loop)
# compile_opt=(dacapo)
library=(HEAAN)
backend=(GPU)
epochs=(10 20 30 40)
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
 hc-test $1 $2 $3 $5 $6 $7 $8 >> $DIR/results/$conf_name/${timestamp}/$4
 # hc-test $1 $2 $3 >> $DIR/results/$conf_name/tmp
 return 0
}
for name in ${bench_name[@]}
do
 for hw in ${backend[@]}
 do
  for ep in ${epochs[@]}
  do
   hc-trace $name $ep $pk
   for wtr in ${waterline[@]}
   do
    for opt in ${compile_opt[@]}
    do
     filename=${name}-${opt}-${wtr}-${ep}.txt
     for (( i = 0; i < 1; i++ ))
     do
      echo "Start compilation of ${name} ( waterline: ${wtr}, opt: ${opt}, ep: ${ep})"
      compile $opt $wtr $name $filename $library $hw $ep
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
      echo "epoch: $ep" >> $DIR/results/$conf_name/${timestamp}/$filename
      echo "compiler: $opt" >> $DIR/results/$conf_name/${timestamp}/$filename
      echo "benchname: $name" >> $DIR/results/$conf_name/${timestamp}/$filename

      echo "Process ${name}_${wtr}_${opt}_${library}_${hw}_${ep}"
      # runner $opt $wtr $name $filename $library $hw $ep $ep2
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

# bench_name=(PCA_Loop)
# bench_name=(PCA)
bench_name=()

waterline=(36)
# waterline=($(seq 30 40))
# compile_opt=(eva)
compile_opt=(dacapo)
# compile_opt=(flex_loop simple_loop)
library=(HEAAN)
backend=(GPU)
epochs=(2 4 6 8)
epoch2=(2 4 6 8)
packing=(0)


for name in ${bench_name[@]}
do
 for hw in ${backend[@]}
 do
  # filename=${name}-${compile_opt}.txt
  for pk in ${packing[@]}
  do
   for ep2 in ${epoch2[@]}
   do
    for ep in ${epochs[@]}
    do
     hc-trace $name $ep $ep2
     for wtr in ${waterline[@]}
     do
      for opt in ${compile_opt[@]}
      do
       filename=${name}-${opt}-${wtr}-${ep}-${ep2}.txt
       for (( i = 0; i < 5; i++ ))
       do
        echo "Start compilation of ${name} ( waterline: ${wtr}, opt: ${opt}, ep: ${ep}, ep: ${ep2})"
        compile $opt $wtr $name $filename $library $hw $ep
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
        echo "packing: ${pk}" >> $DIR/results/$conf_name/${timestamp}/$filename

        echo "Process ${name}_${wtr}_${opt}_${library}_${hw}_${ep}"
        runner $opt $wtr $name $filename $library $hw $ep $ep2
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
 done
done

echo $timestamp
