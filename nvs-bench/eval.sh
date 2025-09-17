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


## To match `full_eval.py` and the bash_scripts, we have to define some additional parameters - max_shapes and indoor_outdoor_flags

# Specify the max number of shapes for each scene
case "$data_folder" in
    */mipnerf360/bicycle)
        max_shapes=6400000
        ;;
    */mipnerf360/flowers)
        max_shapes=5500000
        ;;
    */mipnerf360/garden)
        max_shapes=5200000
        ;;
    */mipnerf360/stump)
        max_shapes=4750000
        ;;
    */mipnerf360/treehill)
        max_shapes=5000000
        ;;
    */mipnerf360/room)
        max_shapes=2100000
        ;;
    */mipnerf360/counter)
        max_shapes=2500000
        ;;
    */mipnerf360/kitchen)
        max_shapes=2400000
        ;;
    */mipnerf360/bonsai)
        max_shapes=3000000
        ;;
    */tanksandtemples/truck)
        max_shapes=2000000
        ;;
    */tanksandtemples/train)
        max_shapes=2500000
        ;;
    *)
        echo "Using default max_shapes=4000000"
        max_shapes=4000000
        ;;
esac

# Check if scene is outdoor or indoor
case "$data_folder" in
    */mipnerf360/bicycle|*/mipnerf360/flowers|*/mipnerf360/garden|*/mipnerf360/stump|*/mipnerf360/treehill|*/tanksandtemples/truck|*/tanksandtemples/train)
        indoor_outdoor_flags="--outdoor"
        ;;
    */mipnerf360/room|*/mipnerf360/counter|*/mipnerf360/kitchen|*/mipnerf360/bonsai|*/deepblending/playroom|*/deepblending/drjohnson|*/zipnerf/nyc|*/zipnerf/london|*/zipnerf/alameda|*/zipnerf/berlin)
        indoor_outdoor_flags="--importance_threshold 0.025 --lr_sigma 0.0008 --opacity_lr 0.014 --lambda_normals 0.00004 --lambda_dist 1 --iteration_mesh 5000 --lambda_opacity 0.0055 --lambda_dssim 0.4 --lr_triangles_points_init 0.0015 --lambda_size 5e-8"
        ;;
esac


python train.py -s $data_folder -m $output_folder --eval --iterations $iterations --max_shapes $max_shapes --quiet --eval --test_iterations -1 $indoor_outdoor_flags
python render.py -s $data_folder -m $output_folder --eval --iteration $iterations --quiet --eval --skip_train
cp -r $output_folder/test/ours_$iterations/renders $output_folder/test_renders

# As a sanity check, print the method's metrics
python metrics.py -m $output_folder