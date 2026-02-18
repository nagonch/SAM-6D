#!/bin/bash

# Base directory for the dataset
BASE_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset"

# Define a mapping of Model Name -> Representative Sequence Name
# This ensures we only run once per unique model.
declare -A model_to_seq
model_to_seq["021_bleach_cleanser"]="bleach_hard_00_03_chaitanya"
model_to_seq["003_cracker_box"]="cracker_box_reorient"
model_to_seq["006_mustard_bottle"]="mustard_easy_00_02"
model_to_seq["004_sugar_box"]="sugar_box_yalehand0"
model_to_seq["005_tomato_soup_can"]="tomato_soup_can_yalehand0"

# Iterate through the unique models
for MODEL_NAME in "${!model_to_seq[@]}"; do
    SEQ_NAME=${model_to_seq[$MODEL_NAME]}
    
    # Construct paths
    CAD_PATH="$BASE_PATH/$SEQ_NAME/frame_0000/obj.ply"
    OUTPUT_DIR="$BASE_PATH/$MODEL_NAME"
    
    echo "---------------------------------------------------"
    echo "Processing Model: $MODEL_NAME"
    echo "Using CAD from:  $CAD_PATH"
    echo "Output Dir:      $OUTPUT_DIR"
    echo "---------------------------------------------------"

    # Ensure output directory exists
    mkdir -p "$OUTPUT_DIR"

    # Run BlenderProc
    blenderproc run render_custom_templates.py \
        --output_dir "$OUTPUT_DIR" \
        --cad_path "$CAD_PATH"
        # --colorize True 
done

echo "Done! All unique models have been processed."