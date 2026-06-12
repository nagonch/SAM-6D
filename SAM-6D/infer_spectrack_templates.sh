#!/bin/bash
# Runs SAM-6D on all SpecTrack sequences.
# Templates must already be rendered with render_spectrack_templates.sh before running this.
# Usage: cd /home/ngoncharov/cvpr2026/SAM-6D/SAM-6D && bash infer_spectrack.sh

# ─── Config ────────────────────────────────────────────────────────────────────
export SEGMENTOR_MODEL=sam
export CUDA_VISIBLE_DEVICES=0

DATASET_ROOT="/home/ngoncharov/SpecTrack_dataset"
MESH_DIR="$DATASET_ROOT/object_meshes"
TEMPLATES_DIR="$(pwd)/results_sam6d/SpecTrack/templates"
BASE_OUTPUT_DIR="$(pwd)/results_sam6d/SpecTrack"

ISM_DIR="$(pwd)/Instance_Segmentation_Model"
PEM_DIR="$(pwd)/Pose_Estimation_Model"

REFLECTIVITIES=("0.0" "0.5" "0.7" "1.0")
DEPTH_MODES=("gt" "synth")

# Center view in a 5x5 light field (0-indexed: row 2, col 2 = index 12)
LF_CENTER="0012.png"

# Maps objects sequence name → mesh folder name (under object_meshes/)
declare -A SEQ_TO_MESH
SEQ_TO_MESH["bleach0"]="bleach_cleanser"
SEQ_TO_MESH["bleach_hard_00_03_chaitanya"]="bleach_cleanser"
SEQ_TO_MESH["cracker_box_reorient"]="cracker_box"
SEQ_TO_MESH["cracker_box_yalehand0"]="cracker_box"
SEQ_TO_MESH["mustard0"]="mustard_bottle"
SEQ_TO_MESH["mustard_easy_00_02"]="mustard_bottle"
SEQ_TO_MESH["sugar_box1"]="sugar_box"
SEQ_TO_MESH["sugar_box_yalehand0"]="sugar_box"
SEQ_TO_MESH["tomato_soup_can_yalehand0"]="tomato_soup_can"

# ─── Tracking ─────────────────────────────────────────────────────────────────
TOTAL_FRAMES=0
DONE_FRAMES=0
SKIPPED_FRAMES=0
FAILED_FRAMES=0
JOB_START=0

# ─── Helper: write camera.json from camera_matrix.txt ─────────────────────────
make_camera_json() {
    local matrix_file="$1"
    local out_json="$2"
    python3 - "$matrix_file" "$out_json" <<'PYEOF'
import sys, json
matrix_file, out_json = sys.argv[1], sys.argv[2]
vals = []
with open(matrix_file) as f:
    for line in f:
        line = line.strip()
        if line:
            vals.extend(float(v) for v in line.split())
with open(out_json, "w") as f:
    json.dump({"cam_K": vals, "depth_scale": 1.0}, f)
PYEOF
}

# ─── Helper: format seconds as HH:MM:SS ───────────────────────────────────────
fmt_time() {
    local secs="$1"
    printf "%02d:%02d:%02d" $((secs/3600)) $(( (secs%3600)/60 )) $((secs%60))
}

# ─── Helper: print progress summary ───────────────────────────────────────────
print_progress() {
    local now; now=$(date +%s)
    local elapsed=$(( now - JOB_START ))
    local total_done=$(( DONE_FRAMES + SKIPPED_FRAMES ))
    local pct=0
    local eta_str="?"
    if [ "$TOTAL_FRAMES" -gt 0 ]; then
        pct=$(( total_done * 100 / TOTAL_FRAMES ))
    fi
    if [ "$total_done" -gt 0 ] && [ "$elapsed" -gt 0 ]; then
        local remaining=$(( (TOTAL_FRAMES - total_done) * elapsed / total_done ))
        eta_str=$(fmt_time "$remaining")
    fi
    local filled=$(( pct / 2 ))
    local empty=$(( 50 - filled ))
    printf -v bar_filled "%${filled}s" ""; printf -v bar_empty "%${empty}s" ""
    local bar="${bar_filled// /#}${bar_empty// /-}"
    echo "  ┌─ Progress ──────────────────────────────────────────────────────┐"
    printf  "  │ [%s] %3d%% │ done=%-5d skipped=%-5d failed=%-4d │ elapsed=%s eta=%s\n" \
        "$bar" "$pct" "$DONE_FRAMES" "$SKIPPED_FRAMES" "$FAILED_FRAMES" \
        "$(fmt_time "$elapsed")" "$eta_str"
    echo "  └─────────────────────────────────────────────────────────────────┘"
}

# ─── Count total work upfront ─────────────────────────────────────────────────
count_total() {
    local total=0
    for dtype in cube objects; do
        for refl in "${REFLECTIVITIES[@]}"; do
            local d="$DATASET_ROOT/${dtype}_${refl}"
            for sp in "$d"/*/; do
                [ -d "$sp" ] || continue
                [ "$(basename "$sp")" = "models" ] && continue
                local n
                n=$(find "$sp" -maxdepth 1 -name "LF_*" -type d | wc -l)
                total=$(( total + n ))
            done
        done
    done
    echo $(( total * ${#DEPTH_MODES[@]} ))
}

# ══════════════════════════════════════════════════════════════════════════════
echo "========================================================================"
echo " SAM-6D inference on SpecTrack dataset"
echo "========================================================================"

# ─── Verify templates exist ───────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo " Checking rendered templates"
echo "──────────────────────────────────────────"
ALL_OBJECTS=("cube" "bleach_cleanser" "cracker_box" "mustard_bottle" "sugar_box" "tomato_soup_can")
MISSING_TEMPLATES=0
for OBJ in "${ALL_OBJECTS[@]}"; do
    TPATH="$TEMPLATES_DIR/$OBJ/templates/xyz_0.npy"
    if [ -f "$TPATH" ]; then
        N=$(ls "$TEMPLATES_DIR/$OBJ/templates/xyz_"*.npy 2>/dev/null | wc -l)
        echo "  $OBJ: $N views"
    else
        echo "  $OBJ: MISSING — run render_spectrack_templates.sh first"
        MISSING_TEMPLATES=$(( MISSING_TEMPLATES + 1 ))
    fi
done
if [ "$MISSING_TEMPLATES" -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_TEMPLATES object(s) have no rendered templates."
    echo "Run: bash render_spectrack_templates.sh"
    exit 1
fi
echo "All templates present."

# ─── Run inference ────────────────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo " STEP 1 / 2  Running inference"
echo "──────────────────────────────────────────"
echo "Counting frames..."
TOTAL_FRAMES=$(count_total)
JOB_START=$(date +%s)
echo "Total frames to process: $TOTAL_FRAMES"
echo ""

for DEPTH_MODE in "${DEPTH_MODES[@]}"; do
    for DATASET_TYPE in cube objects; do
        for REFL in "${REFLECTIVITIES[@]}"; do
            SEQ_DIR="$DATASET_ROOT/${DATASET_TYPE}_${REFL}"

            for SEQ_PATH in "$SEQ_DIR"/*/; do
                [ -d "$SEQ_PATH" ] || continue
                SEQ=$(basename "$SEQ_PATH")
                [ "$SEQ" = "models" ] && continue

                # Determine mesh for this sequence
                if [ "$DATASET_TYPE" = "cube" ]; then
                    MESH_NAME="cube"
                else
                    MESH_NAME="${SEQ_TO_MESH[$SEQ]:-}"
                    if [ -z "$MESH_NAME" ]; then
                        echo "WARNING: unknown sequence '$SEQ', skipping"
                        continue
                    fi
                fi

                CAD_PATH="$MESH_DIR/$MESH_NAME/textured_simple.obj"
                # BlenderProc saves templates inside a templates/ subdirectory
                TEMPLATE_DIR="$TEMPLATES_DIR/$MESH_NAME/templates"

                # Camera intrinsics are the same for every frame in a sequence
                CAMERA_JSON_PATH="/tmp/spectrack_cam_${DATASET_TYPE}_${REFL}_${SEQ}.json"
                make_camera_json "$SEQ_PATH/camera_matrix.txt" "$CAMERA_JSON_PATH"

                # Sorted list of LF frame dirs
                mapfile -t LF_DIRS < <(find "$SEQ_PATH" -maxdepth 1 -name "LF_*" -type d | sort)
                N_FRAMES=${#LF_DIRS[@]}

                echo "┌──────────────────────────────────────────────────────────────┐"
                printf "│ depth=%-5s  dataset=%-10s  refl=%-4s  seq=%-30s │\n" \
                    "$DEPTH_MODE" "$DATASET_TYPE" "$REFL" "$SEQ"
                printf "│ mesh=%-20s  n_frames=%-5d                          │\n" \
                    "$MESH_NAME" "$N_FRAMES"
                echo "└──────────────────────────────────────────────────────────────┘"

                FRAME_IDX=0
                for LF_DIR in "${LF_DIRS[@]}"; do
                    FRAME_IDX=$(( FRAME_IDX + 1 ))
                    FRAME=$(basename "$LF_DIR")
                    FRAME_NUM="${FRAME#LF_}"

                    RGB_PATH="$LF_DIR/$LF_CENTER"
                    if [ "$DEPTH_MODE" = "gt" ]; then
                        DEPTH_PATH="$SEQ_PATH/depth/${FRAME_NUM}.png"
                    else
                        DEPTH_PATH="$SEQ_PATH/depth_synth/${FRAME_NUM}.png"
                    fi

                    OUTPUT_DIR="$BASE_OUTPUT_DIR/$DEPTH_MODE/${DATASET_TYPE}_${REFL}/$SEQ/$FRAME"
                    mkdir -p "$OUTPUT_DIR/sam6d_results"
                    SEG_PATH="$OUTPUT_DIR/sam6d_results/detection_ism.json"
                    PEM_OUT="$OUTPUT_DIR/sam6d_results/detection_pem.json"

                    printf "  [%3d/%3d] %s  " "$FRAME_IDX" "$N_FRAMES" "$FRAME"

                    # Skip if already done
                    if [ -f "$PEM_OUT" ]; then
                        echo "[SKIP - already done]"
                        SKIPPED_FRAMES=$(( SKIPPED_FRAMES + 1 ))
                        continue
                    fi

                    FRAME_OK=true

                    # ── ISM ──────────────────────────────────────────────────
                    echo -n "[ISM..."
                    if ! (cd "$ISM_DIR" && python3 run_inference_custom.py \
                            --segmentor_model "$SEGMENTOR_MODEL" \
                            --output_dir "$OUTPUT_DIR" \
                            --cad_path "$CAD_PATH" \
                            --template_dir "$TEMPLATE_DIR" \
                            --rgb_path "$RGB_PATH" \
                            --depth_path "$DEPTH_PATH" \
                            --cam_path "$CAMERA_JSON_PATH") >> "$OUTPUT_DIR/ism.log" 2>&1; then
                        echo " FAIL]"
                        echo "    ISM error → see $OUTPUT_DIR/ism.log"
                        FRAME_OK=false
                    else
                        echo -n "OK] "
                    fi

                    # ── PEM ──────────────────────────────────────────────────
                    if [ "$FRAME_OK" = true ]; then
                        echo -n "[PEM..."
                        if ! (cd "$PEM_DIR" && python3 run_inference_custom.py \
                                --output_dir "$OUTPUT_DIR" \
                                --cad_path "$CAD_PATH" \
                                --template_path "$TEMPLATE_DIR" \
                                --rgb_path "$RGB_PATH" \
                                --depth_path "$DEPTH_PATH" \
                                --cam_path "$CAMERA_JSON_PATH" \
                                --seg_path "$SEG_PATH") >> "$OUTPUT_DIR/pem.log" 2>&1; then
                            echo " FAIL]"
                            echo "    PEM error → see $OUTPUT_DIR/pem.log"
                            FAILED_FRAMES=$(( FAILED_FRAMES + 1 ))
                        else
                            echo "[OK]"
                            DONE_FRAMES=$(( DONE_FRAMES + 1 ))
                        fi
                    else
                        FAILED_FRAMES=$(( FAILED_FRAMES + 1 ))
                    fi

                done  # LF frames

                print_progress
                echo ""

            done  # sequences
        done  # reflectivities
    done  # dataset types
done  # depth modes

echo ""
echo "========================================================================"
echo " STEP 1 complete"
printf "  total=%d  done=%d  skipped=%d  failed=%d\n" \
    "$TOTAL_FRAMES" "$DONE_FRAMES" "$SKIPPED_FRAMES" "$FAILED_FRAMES"
echo "========================================================================"

# ─── Step 2: Convert to numpy ─────────────────────────────────────────────────
echo ""
echo "──────────────────────────────────────────"
echo " STEP 2 / 2  Converting results to numpy"
echo "──────────────────────────────────────────"
python3 "$(pwd)/convert_spectrack_results.py" --results_dir "$BASE_OUTPUT_DIR"

echo ""
echo "========================================================================"
echo " All done.  Numpy results: $BASE_OUTPUT_DIR/numpy/"
echo "========================================================================"
