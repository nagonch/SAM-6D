# Render CAD templates
export CAD_PATH=/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/Data/Example/obj_000005.ply    # path to a given cad model(mm)
export RGB_PATH=/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/Data/Example/rgb.png           # path to a given RGB image
export DEPTH_PATH=/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/Data/Example/depth.png       # path to a given depth map(mm)
export CAMERA_PATH=/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/Data/Example/camera.json    # path to given camera intrinsics
export OUTPUT_DIR=/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/Data/Example/outputs 

cd Render
blenderproc run render_custom_templates.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH #--colorize True 


# Run instance segmentation model
export SEGMENTOR_MODEL=sam

cd ../Instance_Segmentation_Model
python run_inference_custom.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH


# Run pose estimation model
export SEG_PATH=$OUTPUT_DIR/sam6d_results/detection_ism.json

cd ../Pose_Estimation_Model
python run_inference_custom.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH

