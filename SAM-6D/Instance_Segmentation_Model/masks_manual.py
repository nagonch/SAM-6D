import numpy as np
import json
from pycocotools import mask as maskUtils
from PIL import Image
import os
import numpy as np
import json
from PIL import Image
from pycocotools import mask as maskUtils


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


def convert_mask_to_json(mask_path, scene_id=0, image_id=0, category_id=1, score=1.0):
    binary_mask_orig = np.array(
        Image.open(mask_path).resize((640, 360), resample=Image.NEAREST)
    )
    binary_mask = np.copy(binary_mask_orig)

    binary_mask = (binary_mask > 0).astype(np.uint8)
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    bbox = [int(xmin), int(ymin), int(xmax), int(ymax)]

    output_item = {
        "scene_id": scene_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "score": float(score),
        "time": 0.0,
        "segmentation": mask_to_rle(binary_mask),
    }

    return output_item, binary_mask_orig


if __name__ == "__main__":
    DATASET_INPUT_PATH = "/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/datasets/LiFT_dataset"
    DATASET_OUTPUT_PATH = (
        "/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/input_datasets/LiFT_dataset"
    )
    for SEQUENCE_NAME in [
        "box_motion_prod",
        "car_prod",
        "car_shiny_prod",
        "jug_motion_prod",
        "jug_tilt_prod",
        "jug_translation_z_prod",
        "shiny_box_tilt_prod",
        "teabox_tilt_prod",
        "teabox_translation_prod",
    ]:
        print(SEQUENCE_NAME)
        result_dicts = []
        frame_n = 0
        for folder in sorted(os.listdir(DATASET_INPUT_PATH + "/" + SEQUENCE_NAME)):
            if not folder.startswith("LF_"):
                continue
            mask_path = os.path.join(
                DATASET_INPUT_PATH, SEQUENCE_NAME, folder, "masks/0040.png"
            )
            converted, binary_mask = convert_mask_to_json(mask_path)
            path_out = os.path.join(
                DATASET_OUTPUT_PATH,
                SEQUENCE_NAME,
                f"frame_{str(frame_n).zfill(4)}",
                "sam6d_results",
            )
            os.makedirs(path_out, exist_ok=True)
            with open(os.path.join(path_out, "detection_ism.json"), "w") as f:
                json.dump([converted], f)
            np.save(os.path.join(path_out, "detection_ism.npy"), binary_mask)
            frame_n += 1
            # with open()
            # result_dicts.append(converted)
