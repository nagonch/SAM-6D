#!/bin/bash

# --- Configuration ---
export SEGMENTOR_MODEL=sam
DATASET_ROOT="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/ycbv_lf"
BASE_OUTPUT_DIR="results_sam6d/ycbv_lf"

# Paths to the model directories
ISM_DIR="$(pwd)/Instance_Segmentation_Model"
PEM_DIR="$(pwd)/Pose_Estimation_Model"

# Mapping sequences to models
sequence_names=(
    "bleach_hard_00_03_chaitanya"
    "bleach0"
    "cracker_box_reorient"
    "cracker_box_yalehand0"
    "mustard_easy_00_02"
    "mustard0"
    "sugar_box_yalehand0"
    "sugar_box1"
    "tomato_soup_can_yalehand0"
)

model_names=(
    "021_bleach_cleanser"
    "021_bleach_cleanser"
    "003_cracker_box"
    "003_cracker_box"
    "006_mustard_bottle"
    "006_mustard_bottle"
    "004_sugar_box"
    "004_sugar_box"
    "005_tomato_soup_can"
)

# --- Processing Loop ---
for i in "${!sequence_names[@]}"; do
    SEQ="${sequence_names[$i]}"
    MODEL_NAME="${model_names[$i]}"
    
    SEQ_PATH="$DATASET_ROOT/$SEQ"
    TEMPLATE_DIR="$DATASET_ROOT/$MODEL_NAME"
    
    echo "================================================"
    echo "STARTING SEQUENCE: $SEQ"
    echo "USING TEMPLATES: $MODEL_NAME"
    echo "================================================"

    # Find all frame directories
    for FRAME_DIR_PATH in "$SEQ_PATH"/frame_*; do
        [ -d "$FRAME_DIR_PATH" ] || continue 
        
        FRAME_NAME=$(basename "$FRAME_DIR_PATH")
        
        # Frame-specific inputs
        RGB_PATH="$FRAME_DIR_PATH/rgb.png"
        DEPTH_PATH="$FRAME_DIR_PATH/depth.png"
        CAMERA_PATH="$FRAME_DIR_PATH/camera.json"
        
        # KEY CHANGE: CAD path is the .ply inside the frame folder
        CAD_PATH="$FRAME_DIR_PATH/obj.ply"
        
        # Per-frame output directory
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$SEQ/$FRAME_NAME"
        mkdir -p "$OUTPUT_DIR"

        echo ">>> Processing $FRAME_NAME"

        # 1. Instance Segmentation
        pushd "$ISM_DIR" > /dev/null
        python run_inference_custom.py \
            --segmentor_model "$SEGMENTOR_MODEL" \
            --output_dir "$FRAME_DIR_PATH" \
            --cad_path "$CAD_PATH" \
            --template_dir "$TEMPLATE_DIR" \
            --rgb_path "$RGB_PATH" \
            --depth_path "$DEPTH_PATH" \
            --cam_path "$CAMERA_PATH"
        popd > /dev/null

        # 2. Pose Estimation
        export SEG_PATH="$FRAME_DIR_PATH/sam6d_results/detection_ism.json"

        pushd "$PEM_DIR" > /dev/null
        python run_inference_custom.py \
            --output_dir "$OUTPUT_DIR" \
            --cad_path "$CAD_PATH" \
            --template_path "$TEMPLATE_DIR" \
            --rgb_path "$RGB_PATH" \
            --depth_path "$DEPTH_PATH" \
            --cam_path "$CAMERA_PATH" \
            --seg_path "$SEG_PATH"
        popd > /dev/null
    done
done

echo "Success: All sequences processed."