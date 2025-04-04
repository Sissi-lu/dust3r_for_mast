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

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class ScaredKey(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, running_list, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        assert self.split is not None
        self.mask_bg = mask_bg
        self.dataset_label = 'scared'

        self.pairs = []
        self.scenes = []

        for img_path in sorted(running_list):
            if img_path is not None:
                self.pairs.append([osp.join(img_path, "Left_Image.png"),
                                   osp.join(img_path, "Right_Image.png")])

        self.scene = sorted(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _read_depthmap(self, depth_path):
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_map = (depth_map.astype(float) / 256.0).astype("float32")
        return depth_map

    def _get_views(self, idx, resolution, rng):
        pairs = self.pairs[idx]
        views = []


        parameter_path = str(pairs[0]).replace("Left_Image.png", "calib.json")
        with open(parameter_path, "r") as f:
            calib = json.load(f)

        for idx, one_side in enumerate(pairs):
            rgb_path = one_side
            rgb_image = imread_cv2(rgb_path)

            # camera pose is w2c
            if idx == 0:
                depth_path = str(one_side).replace("Left_Image", "depthmap_left")
                depth_map = self._read_depthmap(depth_path)
                intrinsics = np.array(calib["K_l"], dtype="float32")
                camera_pose = np.linalg.inv(np.array(calib["E_w2l"], dtype="float32"))
            else:
                depth_path = str(one_side).replace("Right_Image", "depthmap_right")
                depth_map = self._read_depthmap(depth_path)
                intrinsics = np.array(calib["K_r"], dtype="float32")
                camera_pose = np.linalg.inv(np.array(calib["E_w2r"], dtype="float32"))

            rgb_image, depth_map, intrinsics = self._crop_resize_if_necessary(rgb_image, depth_map, intrinsics, resolution, rng, info=rgb_path)

            num_valid = (depth_map > 0.0).sum()
            if num_valid <= 0:
                print(rgb_path)
            assert num_valid > 0, "Wrong path : %s"%rgb_path

            frame_num = str(rgb_path).split("/")[-3] + "_" + str(rgb_path).split("/")[-1].split(".")[0]
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
    import matplotlib.pyplot as plt
    import torch
    from tqdm import tqdm

    dataset = ScaredKey(split='train', ROOT="/data/luxiaoxi/dataset/medical_depth/SCARED_keyframe_nearest_256", resolution=224, aug_crop=16)
    # for idx in np.random.permutation(len(dataset)):
    for idx in tqdm(range(len(dataset))):
        views = dataset[idx]
        assert len(views) == 2

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

        print(view_name(views[0]), view_name(views[1]))
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
