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
# This is due to a bug that causes the model's to crash around the 26k iterations mark.
# We attempted to fix it but with limited time were not able to debug the author's code more
# extensively. It seems many datasets are affected.
# The issue seems to stem from aggressive pruning of low-opacity triangles.
#
# Some issues that were tracking this problem:
# https://github.com/trianglesplatting/triangle-splatting/issues/1
# https://github.com/trianglesplatting/triangle-splatting/issues/11
# https://github.com/trianglesplatting/triangle-splatting/issues/28
#
# We recorded our attempts to fix it and eventual failure at:
# https://github.com/trianglesplatting/triangle-splatting/issues/11#issuecomment-3281961127
if [[ $data_folder == *"zipnerf"* ]]; then
    echo "Setting iterations to 25000 for zipnerf datasets due to bug in pruning of low-opacity triangles"
    iterations=25000
fi


## To match `full_eval.py`, we have to define some additional parameters - max_shapes and outdoor

scene_name=$(basename "$data_folder")

# Specify the max number of shapes for each scene
if [[ "$scene_name" == "bicycle" ]]; then
    cap_max=6400000
elif [[ "$scene_name" == "flowers" ]]; then
    cap_max=5500000
elif [[ "$scene_name" == "garden" ]]; then
    cap_max=5200000
elif [[ "$scene_name" == "stump" ]]; then
    cap_max=4750000
elif [[ "$scene_name" == "treehill" ]]; then
    cap_max=5000000
elif [[ "$scene_name" == "room" ]]; then
    cap_max=2100000
elif [[ "$scene_name" == "counter" ]]; then
    cap_max=2500000
elif [[ "$scene_name" == "kitchen" ]]; then
    cap_max=2400000
elif [[ "$scene_name" == "bonsai" ]]; then
    cap_max=3000000
elif [[ "$scene_name" == "truck" ]]; then
    cap_max=2000000
elif [[ "$scene_name" == "train" ]]; then
    cap_max=2500000
else
    echo "Using default cap_max=4000000"
    cap_max=4000000
fi

# Define outdoor scenes
outdoor_scenes=("bicycle" "flowers" "garden" "stump" "treehill" "truck" "train")
indoor_scenes=("room" "counter" "kitchen" "bonsai" "playroom" "drjohnson" "nyc" "london" "alameda" "berlin")

if [[ " ${outdoor_scenes[*]} " =~ " ${scene_name} " ]]; then # check if scene is in outdoor_scenes
    indoor_outdoor_flags="--outdoor"
elif [[ " ${indoor_scenes[*]} " =~ " ${scene_name} " ]]; then # check if scene is in indoor_scenes
    indoor_outdoor_flags="--importance_threshold 0.025 --lr_sigma 0.0008 --opacity_lr 0.014 --lambda_normals 0.00004 --lambda_dist 1 --iteration_mesh 5000 --lambda_opacity 0.0055 --lambda_dssim 0.4 --lr_triangles_points_init 0.0015 --lambda_size 5e-8"
fi


python train.py -s $data_folder -m $output_folder --eval --iterations $iterations --max_shapes $cap_max --quiet --eval --test_iterations -1 $indoor_outdoor_flags
python render.py -s $data_folder -m $output_folder --eval --iteration $iterations --quiet --eval --skip_train
cp -r $output_folder/test/ours_$iterations/renders $output_folder/test_renders

# As a sanity check, print the method's metrics
python metrics.py -m $output_folder