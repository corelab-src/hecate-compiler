#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# output_file="$DIR/total.txt"
CONFERENCE="ASPLOS25"
output_file="$DIR/$CONFERENCE/pca.txt"
touch $output_file
mkdir "$DIR/$CONFERENCE/csv"
mkdir "$DIR/$CONFERENCE/pdf"

# 11_15_003548 #elasm
DIRECTORIES=(
 "$CONFERENCE/PCA_result" #regression except unroll loop
 "$CONFERENCE/2024_06_18_153021" #regression except unroll loop
 "$CONFERENCE/2024_06_19_040206" #pca dacapo
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
