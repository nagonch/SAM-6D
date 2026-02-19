import viser
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
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
            "mask": object_mask.astype(bool),
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
            "mask": object_mask.astype(bool),
            "depth": depth_image,
            "pose": object_pose.astype(np.float32),
        }


def create_viser_server() -> viser.ViserServer:
    server = viser.ViserServer(verbose=False)

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True

    return server


def backproject_depth_to_pointcloud(depths, camera_matrix, return_scales=False):
    depths = torch.from_numpy(depths).double().cuda()
    camera_matrix = torch.from_numpy(camera_matrix).double().cuda()
    uu, vv = torch.meshgrid(
        (
            torch.arange(depths.shape[0], device=depths.device),
            torch.arange(depths.shape[1], device=depths.device),
        )
    )
    uu, vv = uu.reshape(-1), vv.reshape(-1)
    pixel_indices = torch.stack((vv, uu), dim=0).T
    depths = depths.reshape(-1)
    inv_camera_matrix = torch.linalg.inv(camera_matrix).double()
    ones = torch.ones(
        (pixel_indices.shape[0], 1),
        device=pixel_indices.device,
        dtype=pixel_indices.dtype,
    ).double()
    uv1 = torch.cat([pixel_indices, ones], dim=1).T  # Shape: [3, N]
    xyz_camera = (inv_camera_matrix @ uv1) * depths
    xyz_camera = xyz_camera.T
    if return_scales:
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        scales_x = xyz_camera[:, 2] / fx
        scales_y = xyz_camera[:, 2] / fy
        scales = torch.zeros(xyz_camera.shape[0], 3, device=xyz_camera.device)
        scales[:, 0] = scales_x
        scales[:, 1] = scales_y
        scales[:, 2] = (scales_x + scales_y) / 2
        return xyz_camera, scales
    else:
        return xyz_camera.cpu().numpy()


def run_viser_server(server: viser.ViserServer):
    try:
        while True:
            time.sleep(2.0)
    except KeyboardInterrupt:
        server.scene.reset


class Visualizer:
    def __init__(self):
        self.server = create_viser_server()
        self.scene = self.server.scene

    def run(self):
        run_viser_server(self.server)

    def add_point_cloud(self, name, points, colors=None, point_size=1e-4):
        if colors is None:
            colors = np.array([255, 0, 0])
        self.scene.add_point_cloud(name, points, colors=colors, point_size=point_size)

    def add_frame(self, name, frame_T, frames_scale=0.05):
        if not isinstance(frame_T, np.ndarray):
            frame_T = np.array(frame_T)
        position = frame_T[:3, 3]
        rotation = frame_T[:3, :3]
        xyzw = R.from_matrix(rotation).as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
        self.scene.add_frame(
            name=name,
            position=position,
            wxyz=wxyz,
            axes_length=frames_scale * 2,
            origin_radius=frames_scale / 5,
            axes_radius=frames_scale / 10,
        )

    def add_camera_frustum(self, name, camera_T, camera_matrix, image, scale=0.1):
        image_height, image_width = image.shape[:2]
        if not isinstance(camera_T, np.ndarray):
            camera_T = np.array(camera_T)
        position = camera_T[:3, 3]
        rotation = camera_T[:3, :3]
        wxyz = R.from_matrix(rotation).as_quat(scalar_first=True)
        fov = np.arctan2(image_width / 2, camera_matrix[0, 0]) * 2
        self.scene.add_camera_frustum(
            name=name,
            aspect=image_width / image_height,
            fov=fov.item(),
            scale=scale,
            line_width=0.5,
            image=image,
            wxyz=wxyz,
            position=position,
        )

    def add_mesh(self, name, mesh, pose):
        vertices = mesh.verts_list()[0].cpu().numpy()
        faces = mesh.faces_list()[0].cpu().numpy()
        position = pose[:3, 3].cpu().numpy()
        rotation = pose[:3, :3].cpu().numpy()
        wxyz = R.from_matrix(rotation).as_quat(scalar_first=True)
        self.scene.add_mesh_simple(
            name,
            vertices=vertices,
            faces=faces,
            wxyz=wxyz,
            position=position,
        )


if __name__ == "__main__":
    dataset = YCBV_LF(
        "/home/ngoncharov/cvpr2026/megapose6d/datasets/ycbv_lf/bleach_hard_00_03_chaitanya"
    )
    poses = np.load(
        "/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/Pose_Estimation_Model/results_sam6d/ycbv_lf/bleach_hard_00_03_chaitanya.npy"
    )
    visulizer = Visualizer()
    for i in np.linspace(0, len(dataset) - 1, 10).astype(int):
        sample = dataset[i]
        depth = sample["depth"]
        mask = sample["mask"].astype(bool)
        camera_matrix = dataset.camera_intrinsics
        pose_gt = sample["pose"]
        pose = poses[i]
        color = sample["rgb"].reshape(-1, 3)
        points_camera = backproject_depth_to_pointcloud(depth, camera_matrix)

        color = color[mask.reshape(-1)]
        points_camera = points_camera[mask.reshape(-1)]
        points_camera_np = points_camera
        visulizer.add_point_cloud(
            name=f"pointcloud_{i}",
            points=points_camera_np,
            colors=color,
            point_size=1e-3,
        )
        visulizer.add_frame(
            name=f"frame_{i}",
            frame_T=pose,
        )
        visulizer.add_frame(
            name=f"frame_gt_{i}",
            frame_T=pose_gt,
        )
    visulizer.run()
    visualizer = Visualizer()
    visualizer.run()
