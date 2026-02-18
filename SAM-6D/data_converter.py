import numpy as np
import os
import trimesh
from PIL import Image

sequence_names = [
    "bleach_hard_00_03_chaitanya",
    "bleach0",
    "cracker_box_reorient",
    "cracker_box_yalehand0",
    "mustard_easy_00_02",
    "mustard0",
    "sugar_box_yalehand0",
    "sugar_box1",
    "tomato_soup_can_yalehand0",
]
model_names = [
    "021_bleach_cleanser",
    "021_bleach_cleanser",
    "003_cracker_box",
    "003_cracker_box",
    "006_mustard_bottle",
    "006_mustard_bottle",
    "004_sugar_box",
    "004_sugar_box",
    "005_tomato_soup_can",
]

sequence_to_model = {seq: model for seq, model in zip(sequence_names, model_names)}

sequence_names = [
    "bleach_hard_00_03_chaitanya",
    "bleach0",
    "cracker_box_reorient",
    "cracker_box_yalehand0",
    "mustard_easy_00_02",
    "mustard0",
    "sugar_box_yalehand0",
    "sugar_box1",
    "tomato_soup_can_yalehand0",
]
model_names = [
    "021_bleach_cleanser",
    "021_bleach_cleanser",
    "003_cracker_box",
    "003_cracker_box",
    "006_mustard_bottle",
    "006_mustard_bottle",
    "004_sugar_box",
    "004_sugar_box",
    "005_tomato_soup_can",
]

sequence_to_model = {seq: model for seq, model in zip(sequence_names, model_names)}

sequence_names_lift = [
    "box_motion_prod",
    "car_prod",
    "car_shiny_prod",
    "jug_motion_prod",
    "jug_tilt_prod",
    "jug_translation_z_prod",
    "shiny_box_tilt_prod",
    "teabox_tilt_prod",
    "teabox_translation_prod",
]
model_names_lift = [
    "box_ref_prod",
    "car_ref_prod",
    "car_shiny_ref_prod",
    "jug_ref_prod",
    "jug_ref_prod",
    "jug_ref_prod",
    "shiny_box_ref_prod",
    "teabox_ref_prod",
    "teabox_ref_prod",
]

sequence_to_model = {seq: model for seq, model in zip(sequence_names, model_names)}
sequence_to_model_lift = {
    seq: model for seq, model in zip(sequence_names_lift, model_names_lift)
}


class EOAT:
    def __init__(self, data_path):
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        self.data_path = data_path
        self.sequence_name = data_path.split("/")[-1]
        self.model_name = sequence_to_model.get(self.sequence_name, None)
        self.model_path = os.path.join(
            "/".join(data_path.split("/")[:-1]),
            "models",
            self.model_name,
            "textured.obj",
        )
        self.gt_mesh = trimesh.load(self.model_path)

        self.camera_intrinsics = np.loadtxt(os.path.join(self.data_path, "cam_K.txt"))
        self.rgb_path = os.path.join(self.data_path, "rgb")
        self.depth_path = os.path.join(self.data_path, "depth")
        self.poses_path = os.path.join(self.data_path, "annotated_poses")
        self.masks_path = os.path.join(self.data_path, "gt_mask")

        self.rgb_frames = [
            os.path.join(self.rgb_path, f) for f in sorted(os.listdir(self.rgb_path))
        ]
        self.depth_frames = [
            os.path.join(self.depth_path, f)
            for f in sorted(os.listdir(self.depth_path))
        ]
        self.pose_frames = [
            os.path.join(self.poses_path, f)
            for f in sorted(os.listdir(self.poses_path))
        ]
        self.mask_frames = [
            os.path.join(self.masks_path, f)
            for f in sorted(os.listdir(self.masks_path))
        ]

    def __len__(self):
        return len(self.rgb_frames)

    def __getitem__(self, idx):
        rgb = np.array(Image.open(self.rgb_frames[idx]).convert("RGB"))
        depth = np.array(Image.open(self.depth_frames[idx])).astype(np.float32) / 1000.0
        pose = np.loadtxt(self.pose_frames[idx])
        mask = np.array(Image.open(self.mask_frames[idx])).astype(bool)

        return {
            "rgb": rgb,
            "depth": depth,
            "pose": pose,
            "mask": mask,
        }


class YCBV_LF:
    def __init__(self, data_path):
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        self.data_path = data_path
        self.sequence_name = data_path.split("/")[-1]
        self.model_name = sequence_to_model.get(self.sequence_name, None)
        self.model_path = os.path.join(
            "/".join(data_path.split("/")[:-1]),
            "models",
            self.model_name,
            "textured.obj",
        )
        self.gt_mesh = trimesh.load(self.model_path)
        self.model_name = sequence_to_model[self.sequence_name]
        self.gt_mesh = trimesh.load(self.model_path)

        self.camera_poses_paths = [
            os.path.join(self.data_path, "camera_poses", item)
            for item in list(
                sorted(os.listdir(os.path.join(self.data_path, "camera_poses")))
            )
        ]
        self.n_cameras = len(self.camera_poses_paths)
        self.camera_pose = np.loadtxt(self.camera_poses_paths[self.n_cameras // 2])
        self.camera_intrinsics = np.loadtxt(
            os.path.join(self.data_path, "camera_matrix.txt")
        )
        self.depth_dir = os.path.join(self.data_path, "depth")
        self.depth_paths = [
            os.path.join(self.depth_dir, item)
            for item in list(sorted(os.listdir(self.depth_dir)))
        ]
        self.object_poses_dir = os.path.join(self.data_path, "object_poses")
        self.object_poses_paths = [
            os.path.join(self.object_poses_dir, item)
            for item in list(sorted(os.listdir(self.object_poses_dir)))
        ]
        self.lf_paths = [
            os.path.join(self.data_path, item)
            for item in list(sorted(os.listdir(os.path.join(self.data_path))))
            if "LF_" in item
        ]

    def __len__(self):
        return len(self.lf_paths)

    def __getitem__(self, idx):
        lf_path = self.lf_paths[idx]
        depth_path = self.depth_paths[idx]
        object_pose_path = self.object_poses_paths[idx]
        rgb_image = np.array(
            Image.open(f"{lf_path}/{self.n_cameras//2:04d}.png")
        ).astype(np.uint8)
        object_mask = np.array(
            Image.open(f"{lf_path}/masks/{self.n_cameras//2:04d}.png")
        ).astype(np.uint8)
        depth_image = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        object_pose = np.loadtxt(object_pose_path)
        return {
            "rgb": rgb_image,
            "mask": object_mask.astype(np.bool),
            "depth": depth_image,
            "pose": object_pose.astype(np.float32),
        }


class LIFT:
    def __init__(self, data_path, models_path=None):
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."
        self.data_path = data_path
        self.sequence_name = data_path.split("/")[-1]
        self.models_path = models_path
        if self.models_path is not None:
            self.model_path = os.path.join(
                self.models_path,
                sequence_to_model_lift[self.sequence_name],
                "model.obj",
            )
            self.gt_mesh = trimesh.load(self.model_path)
        self.camera_poses_paths = [
            os.path.join(self.data_path, "camera_poses", item)
            for item in list(
                sorted(os.listdir(os.path.join(self.data_path, "camera_poses")))
            )
        ]
        self.n_cameras = len(self.camera_poses_paths)
        self.camera_pose = np.loadtxt(self.camera_poses_paths[self.n_cameras // 2])
        self.camera_intrinsics = np.loadtxt(
            os.path.join(self.data_path, "camera_matrix.txt")
        )
        self.depth_dir = os.path.join(self.data_path, "depth")
        self.depth_paths = [
            os.path.join(self.depth_dir, item)
            for item in list(sorted(os.listdir(self.depth_dir)))
        ]
        self.object_poses_dir = os.path.join(self.data_path, "object_poses")
        self.object_poses_paths = [
            os.path.join(self.object_poses_dir, item)
            for item in list(sorted(os.listdir(self.object_poses_dir)))
        ]
        self.lf_paths = [
            os.path.join(self.data_path, item)
            for item in list(sorted(os.listdir(os.path.join(self.data_path))))
            if "LF_" in item
        ]

    def __len__(self):
        return len(self.lf_paths)

    def __getitem__(self, idx):
        lf_path = self.lf_paths[idx]
        depth_path = self.depth_paths[idx]
        object_pose_path = self.object_poses_paths[idx]
        rgb_image = np.array(
            Image.open(f"{lf_path}/{self.n_cameras//2:04d}.png")
        ).astype(np.uint8)
        object_mask = np.array(
            Image.open(f"{lf_path}/masks/{self.n_cameras//2:04d}.png")
        ).astype(np.uint8)
        depth_image = np.array(Image.open(depth_path), dtype=np.float32) / 1000.0
        object_pose = np.loadtxt(object_pose_path)
        object_pose = np.linalg.inv(self.camera_pose) @ object_pose
        return {
            "rgb": rgb_image,
            "mask": object_mask.astype(np.bool),
            "depth": depth_image,
            "pose": object_pose.astype(np.float32),
        }


if __name__ == "__main__":
    dataset = EOAT(
        "/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/datasets/ycb_in_eoat/bleach_hard_00_03_chaitanya"
    )
    print(dataset[0])
