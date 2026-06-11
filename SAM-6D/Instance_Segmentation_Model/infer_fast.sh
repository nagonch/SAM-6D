SEGMENTOR_MODEL="sam"

FRAME_DIR_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset/box_motion_prod/frame_0000"
CAD_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset/box_motion_prod/frame_0000/obj.ply"
TEMPLATE_DIR="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset/box_ref_prod"
RGB_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset/box_motion_prod/frame_0000/rgb.png"
DEPTH_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset/box_motion_prod/frame_0000/depth.png"
CAMERA_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset/box_motion_prod/frame_0000/camera.json"

# FRAME_DIR_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/ycbv_eoat/bleach0/frame_0000"
# CAD_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/ycbv_eoat/bleach0/frame_0000/obj.ply"
# TEMPLATE_DIR="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/ycbv_eoat/021_bleach_cleanser"
# RGB_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/ycbv_eoat/bleach0/frame_0000/rgb.png"
# DEPTH_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/ycbv_eoat/bleach0/frame_0000/depth.png"
# CAMERA_PATH="/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/ycbv_eoat/bleach0/frame_0000/camera.json"

python run_inference_custom.py \
            --segmentor_model "$SEGMENTOR_MODEL" \
            --output_dir "$FRAME_DIR_PATH" \
            --cad_path "$CAD_PATH" \
            --template_dir "$TEMPLATE_DIR" \
            --rgb_path "$RGB_PATH" \
            --depth_path "$DEPTH_PATH" \
            --cam_path "$CAMERA_PATH"