#!/bin/bash

# Check if scene argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset/scene>"
    echo "Example: $0 mipnerf360/bicycle"
    exit 1
fi

scene=$1

method="triangle-splatting"
expected_output_folder="/nvs-leaderboard-output/$scene/$method/renders_test"

# Remove the output folder if it already exists
rm -rf /nvs-leaderboard-output/$scene/$method

# Record start time
start_time=$(date +%s)

######## START OF YOUR CODE ########
# Train using the train split in the dataset folder
# eg: python train.py --data /nvs-leaderboard-data/$scene/train --output /nvs-leaderboard-output/$scene/$method/

python train.py -s /nvs-leaderboard-data/$scene/train -m /nvs-leaderboard-output/$scene/$method/

# Render the test split
python render.py -s /nvs-leaderboard-data/$scene/test -m /nvs-leaderboard-output/$scene/$method/

# At the end, move your renders into the `expected_output_folder`
# Note: we move them out of the "train" output folder because that's what the gausian splatting pipeline thinks the "test" split is
# during the rendering step.
mv /nvs-leaderboard-output/$scene/$method/train/ours_$iterations/renders $expected_output_folder
######## END OF YOUR CODE ########

# Record end time and show duration
end_time=$(date +%s)
echo $((end_time - start_time)) > /nvs-leaderboard-output/$scene/$method/training_time.txt
