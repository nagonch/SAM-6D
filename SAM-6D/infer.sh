#!/bin/bash

# --- Configuration ---
export SEGMENTOR_MODEL=sam
DATASET_ROOT="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset"
BASE_OUTPUT_DIR="results_sam6d/LiFT_dataset"
CUDA_VISIBLE_DEVICES=1
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
    "bleach_cleanser"
    "bleach_cleanser"
    "cracker_box"
    "cracker_box"
    "mustard_bottle"
    "mustard_bottle"
    "sugar_box"
    "sugar_box"
    "tomato_soup_can"
)


declare -a runtimes=()
total_runtime=0
run_count=0

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
    echo "Sequence Path: $SEQ_PATH"

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
        # python run_inference_custom.py \
        #     --segmentor_model "$SEGMENTOR_MODEL" \
        #     --output_dir "$FRAME_DIR_PATH" \
        #     --cad_path "$CAD_PATH" \
        #     --template_dir "$TEMPLATE_DIR" \
        #     --rgb_path "$RGB_PATH" \
        #     --depth_path "$DEPTH_PATH" \
        #     --cam_path "$CAMERA_PATH"
        popd > /dev/null

        # 2. Pose Estimation
        export SEG_PATH="$FRAME_DIR_PATH/sam6d_results/detection_ism.json"

        pushd "$PEM_DIR" > /dev/null
        start_time=$(date +%s.%N)
        python run_inference_custom.py \
            --output_dir "$OUTPUT_DIR" \
            --cad_path "$CAD_PATH" \
            --template_path "$TEMPLATE_DIR" \
            --rgb_path "$RGB_PATH" \
            --depth_path "$DEPTH_PATH" \
            --cam_path "$CAMERA_PATH" \
            --seg_path "$SEG_PATH"
        end_time=$(date +%s.%N)
        runtime=$(echo "$end_time - $start_time" | bc)
        runtimes+=("$runtime")
        total_runtime=$(echo "$total_runtime + $runtime" | bc)
        ((run_count++))
        popd > /dev/null
    done
done

echo "Success: All sequences processed."
echo "========================================"
echo "All runs finished."
echo "Total runs: $run_count"

if [ "$run_count" -gt 0 ]; then
    avg_runtime=$(echo "$total_runtime / $run_count" | bc -l)
    echo "Average runtime: $avg_runtime sec"
else
    echo "No runs executed."
fi