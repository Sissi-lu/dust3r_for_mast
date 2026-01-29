# --------------------------------------------------------
# Dataloader for preprocessed Scared dataset
# See datasets_preprocess/preprocess_scared.py
# --------------------------------------------------------
import os.path as osp
import json
import itertools
from collections import deque
import os
import cv2
import numpy as np


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Abs(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, running_list, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        assert self.split is not None
        self.mask_bg = mask_bg
        self.dataset_label = 'Abs'

        self.pairs = []
        # self.scenes = []
        for img_path in sorted(running_list):
            if img_path is not None:
                self.pairs.append([img_path, img_path])

        self.scene = sorted(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _read_depthmap(self, depth_path):

        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_map = (depth_map.astype(float) / 256).astype("float32")
        return depth_map

    def _get_views(self, idx, resolution, rng):
        pair_dir = self.pairs[idx][0]
        views = []

        parameter_path = os.path.join(pair_dir, 'calib.json')
        with open(parameter_path, "r") as f:
            calib = json.load(f)
        sides = ["L", "R"]
        for idx, side in enumerate(sides):
            rgb_path = os.path.join(pair_dir, "img%s.png"%side)
            rgb_image = imread_cv2(rgb_path)

            depth_path = os.path.join(pair_dir, "depth%s.png"%side)
            depth_map = self._read_depthmap(depth_path)

            # camera pose is
            if idx == 0:
                intrinsics = np.array(calib["M_l"], dtype="float32")
                camera_pose = np.eye(4, dtype="float32")
            else:
                intrinsics = np.array(calib["M_r"], dtype="float32")
                w2c = np.eye(4)
                w2c[:3, :3] = np.array(calib["R"], dtype="float32")
                w2c[:3, 3] = np.array(calib["T"], dtype="float32").reshape(-1)
                # camera_pose = w2c.astype("float32")
                camera_pose = np.linalg.inv(w2c).astype("float32")

            rgb_image, depth_map, intrinsics = self._crop_resize_if_necessary(rgb_image, depth_map, intrinsics, resolution, rng, info=rgb_path)

            num_valid = (depth_map > 0.0).sum()
            assert num_valid > 0

            frame_num = pair_dir.split("/")[-2] + "_" + pair_dir.split("/")[-1]
            views.append(dict(
                img=rgb_image,
                depthmap=depth_map,
                camera_pose=camera_pose,   # c2w
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=rgb_path,
                instance=frame_num,
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb
    import torch
    import matplotlib.pyplot as plt

    dataset = Abs(split='test', ROOT="/data/luxiaoxi/dataset/medical_depth/EndoAbs_preprocessed_nearest", running_list="sound", resolution=224, aug_crop=16)

    # for idx in np.random.permutation(len(dataset)):
    for idx in range(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))

        depth_left = views[0]["depthmap"]
        depth_right = views[1]["depthmap"]
        rgb_left = views[0]["img"].permute(1, 2, 0)
        rgb_right = views[1]["img"].permute(1, 2, 0)

        depth = np.concatenate([depth_left, depth_right], axis=1)
        plt.imshow(depth, cmap="jet")
        plt.show()

        rgb_combine = torch.concat([rgb_left, rgb_right], dim=1)
        plt.imshow(rgb_combine)
        plt.show()

        diff = rgb_left - rgb_right
        plt.imshow(diff, cmap="jet")
        plt.show()

        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 5)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx] ['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()

