import numpy as np
import json
import os

if __name__ == "__main__":
    predictions_path = "/home/ngoncharov/cvpr2026/SAM-6D/SAM-6D/Pose_Estimation_Model/results_sam6d/ycbv_lf"
    for filename in sorted(os.listdir(predictions_path)):
        if not os.path.isdir(os.path.join(predictions_path, filename)):
            continue
        sequence_path = os.path.join(predictions_path, filename)
        predictions = []
        for frame in sorted(os.listdir(sequence_path)):
            frame_path = os.path.join(
                sequence_path, frame, "sam6d_results/detection_pem.json"
            )
            with open(frame_path, "r") as f:
                pred = json.load(f)
                scores = [item["score"] for item in pred]
                highest_score_idx = np.argmax(scores)
                pred = pred[highest_score_idx]
                R = pred["R"]
                t = pred["t"]
                pose = np.eye(4)
                pose[:3, :3] = R
                pose[:3, 3] = np.array(t) / 1000.0
                predictions.append(pose)
        predictions = np.stack(predictions)
        np.save(f"{predictions_path}/{filename}.npy", predictions)
