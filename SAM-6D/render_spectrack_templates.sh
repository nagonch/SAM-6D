#!/bin/bash
# Renders SAM-6D NOCS templates for all SpecTrack objects using BlenderProc.
# Run once from inside the container:
#   cd /home/ngoncharov/cvpr2026/SAM-6D/SAM-6D && bash render_spectrack_templates.sh
#
# Output: results_sam6d/SpecTrack/templates/{object_name}/templates/{rgb,mask,xyz}_N.*

MESH_DIR="/home/ngoncharov/SpecTrack_dataset/object_meshes"
TEMPLATES_DIR="$(pwd)/results_sam6d/SpecTrack/templates"
RENDER_DIR="$(pwd)/Render"

ALL_OBJECTS=("cube" "bleach_cleanser" "cracker_box" "mustard_bottle" "sugar_box" "tomato_soup_can")

mkdir -p "$TEMPLATES_DIR"

TOTAL=${#ALL_OBJECTS[@]}
IDX=0

for OBJ in "${ALL_OBJECTS[@]}"; do
    IDX=$(( IDX + 1 ))
    CAD_PATH="$MESH_DIR/$OBJ/textured_simple.obj"
    TEMPLATE_OUT="$TEMPLATES_DIR/$OBJ"

    if [ ! -f "$CAD_PATH" ]; then
        echo "[$IDX/$TOTAL] $OBJ: mesh not found at $CAD_PATH, skipping"
        continue
    fi

    # Templates land at $TEMPLATE_OUT/templates/xyz_0.npy
    if [ -f "$TEMPLATE_OUT/templates/xyz_0.npy" ]; then
        echo "[$IDX/$TOTAL] $OBJ: already rendered, skipping"
        continue
    fi

    echo "[$IDX/$TOTAL] $OBJ: rendering ..."
    mkdir -p "$TEMPLATE_OUT"
    (cd "$RENDER_DIR" && blenderproc run render_custom_templates.py \
        --output_dir "$TEMPLATE_OUT" \
        --cad_path "$CAD_PATH")

    if [ -f "$TEMPLATE_OUT/templates/xyz_0.npy" ]; then
        N=$(ls "$TEMPLATE_OUT/templates/xyz_"*.npy 2>/dev/null | wc -l)
        echo "[$IDX/$TOTAL] $OBJ: done ($N views)"
    else
        echo "[$IDX/$TOTAL] $OBJ: FAILED (no output)"
    fi
done

echo ""
echo "========================================"
echo " Templates ready: $TEMPLATES_DIR"
echo "========================================"
