#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# output_file="$DIR/total.txt"
CONFERENCE="ASPLOS25"
output_file="$DIR/$CONFERENCE/total.txt"
touch $output_file
mkdir "$DIR/$CONFERENCE/csv"
mkdir "$DIR/$CONFERENCE/pdf"

# 11_15_003548 #elasm
DIRECTORIES=(
 # "$CONFERENCE/2024_05_31_133811"
 # "$CONFERENCE/2024_05_31_174758"
 # "$CONFERENCE/2024_06_01_172456"
 # "$CONFERENCE/2024_06_04_175028"

 "$CONFERENCE/2024_06_08_171302" #dacapo unpack
 "$CONFERENCE/2024_06_09_031937" #dacapo unpack
 # "$CONFERENCE/2024_06_09_161144" #simple_loop
 # "$CONFERENCE/2024_06_09_234544" #simple_loop PCA
 # "$CONFERENCE/2024_06_10_090024" #simple_loop Multivaraite
 # "$CONFERENCE/2024_06_10_172202" #simple_loop packing LR PR
 # "$CONFERENCE/2024_06_10_180723" #dacapo_flex
 # "$CONFERENCE/2024_06_10_180723" #dacapo_flex
 "$CONFERENCE/2024_06_11_085922" #SVM dacapo
 # "$CONFERENCE/2024_06_11_164912" #LogReg dacapo_flex
 # "$CONFERENCE/2024_06_11_174321" #SVM simple loop unpacking
 "$CONFERENCE/2024_06_11_185042" #Kmeans dacapo unpacking
 # "$CONFERENCE/2024_06_11_190203" #SVM dacapo_flex unpacking
 "$CONFERENCE/2024_06_11_195132" #Polynomial dacapo unpacking
 # "$CONFERENCE/2024_06_11_200238" #Logistic dacapo unpacking
 #######################################################3
 # "$CONFERENCE/2024_06_17_125809" #Logistic loop
 "$CONFERENCE/2024_06_17_103736" #Regression unroll loop
 "$CONFERENCE/2024_06_17_085939" #Kmeans epoch 40
 "$CONFERENCE/2024_06_17_084631" #Kmeans halo epoch 30
 "$CONFERENCE/2024_06_17_014737" #SVM Kmeans loop
 "$CONFERENCE/2024_06_16_165226" #regression except unroll loop
 "$CONFERENCE/2024_06_18_121243" #logistic regression loop
 "$CONFERENCE/2024_06_18_142307" #logistic regression dacapo
)

[ -e "$output_file" ] && rm "$output_file"


for directory in "${DIRECTORIES[@]}"; do
 if [ -d "$directory" ]; then
  cd "$directory" || continue
  for file in *; do
   if [ -e "$file" ]; then
    cat "$file" >> "$output_file"
    echo "File '$file' appended to '$output_file'"
   fi
  done
  cd -
 else
  echo "Warning: dir '$directory' not found. Skipping."
 fi
done

# fine-grained

# ddir="2023_11_15_023502"
ddir=()
files=(
 # "HarrisCornerDetection-optimal.txt"
 # "MLP-optimal.txt"
 # "SobelFilter-optimal.txt"
 # "LinearRegression-optimal.txt"
)

if [ -d "$ddir" ]; then
 cd "$ddir" || exit
 for file in "${files[@]}"; do
  if [ -e "$file" ]; then
   cat "$file" >> "$output_file"
   echo "File '$file' appended to '$output_file'"
  fi
 done
 cd -
else
 echo "Warning: File '$file' not found. Skipping."
fi


echo "Combining files completed. Result saved in '$output_file'"
