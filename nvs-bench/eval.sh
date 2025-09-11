#!/bin/bash
set -e

# Check if data_folder and output_folder arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <data_folder> <output_folder>"
    echo "Example: $0 /nvs-bench/data/mipnerf360/bicycle /nvs-bench/methods/3dgs/mipnerf360/bicycle"
    exit 1
fi
data_folder=$1
output_folder=$2

######## START OF YOUR CODE ########
# 1) Train 
#   python train.py --data $data_folder --output $output_folder --eval
# 2) Render the test split
#   python render.py --data $data_folder/test --output $output_folder --eval
# 3) Move the renders into `$output_folder/test_renders`
#   mv $output_folder/test/ours_30000/renders $output_folder/test_renders

iterations=30000

# If the data folder contains zipnerf then set iterations to 25000
# This is due to a bug that causes the model's to crash around the 27k iterations mark.
# We attempted to fix it but with limited time were not able to debug the author's code more
# extensively. It seems many datasets are affected.
# Some issues that were tracking this problem:
# https://github.com/trianglesplatting/triangle-splatting/issues/1
# https://github.com/trianglesplatting/triangle-splatting/issues/11
# https://github.com/trianglesplatting/triangle-splatting/issues/28
#
# We recorded our attempts to fix it and eventual failure at:
# https://github.com/trianglesplatting/triangle-splatting/issues/11#issuecomment-3281961127
if [[ $data_folder == *"zipnerf"* ]]; then
    echo "Setting iterations to 25000 for zipnerf datasets due to a size bug"
    iterations=25000
fi

python train.py -s $data_folder -m $output_folder --eval --iterations $iterations
python render.py -s $data_folder -m $output_folder --eval --iteration $iterations
mv $output_folder/test/ours_$iterations/renders $output_folder/test_renders