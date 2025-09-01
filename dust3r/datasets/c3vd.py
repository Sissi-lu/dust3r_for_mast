# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# See datasets_preprocess/preprocess_co3d.py
# --------------------------------------------------------
import glob
import os.path as osp
import json
import itertools
from collections import deque
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class C3VD(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'C3VD'
        # load all scenes


        scenes = []
        if self.split == "train":
            with open(osp.join(self.ROOT, f'train.txt'), 'r') as f:
                scenes = f.readline().strip().split(' ')
        elif self.split == "test":
            with open(osp.join(self.ROOT, f'val.txt'), 'r') as f:
                scenes = f.readline().strip().split(' ')


        # scene has the under_review, which is not included
        self.scenes = {(scene, scene): sorted(glob.glob(os.path.join(self.ROOT, scene, "images", "*.png"))) for scene in scenes}

        self.scene_list = list(self.scenes.keys())

        # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
        # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
        # self.combinations = [(i, j)
        #                      for i, j in itertools.combinations(range(60), 2)
        #                      if 0 < abs(i - j) <= 10 and abs(i - j) % 3 == 0 and abs(i - j) != 0]
        self.combinations = [(i, i + k)
                             for i in range(len(self.scenes))
                             for k in [1, 2, 3]
                             if i + k < len(self.scenes)]
        self.invalidate = {scene: {} for scene in self.scene_list}

        self.min_depth = 0.001
        self.max_depth = 100

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)

    def _read_depthmap(self, depthpath):
        depth = np.array(cv2.imread(depthpath, -1))

        # set invalid or clipped depth values as NaN
        # depth = np.where((depth == 0) | (depth == 2 ** 16 - 1), np.nan, depth)

        # convert depth to float and scale it
        depth = (depth.astype(np.float32) / (2 ** 16 - 1)) * 100 # unit: mm
        return depth

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]

        # add a bit of randomness
        last = len(image_pool) - 1

        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            if self.invalidate[obj, instance][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            impath = image_pool[im_idx]
            num = int(impath.split('/')[-1].split('_')[0])
            abs_path = os.path.abspath(os.path.join(impath, "../.."))
            depthpath = os.path.join(abs_path, "depths" ,"%04d_depth.tiff"%num)

            # intrinsic: load camera params
            # cx = 678.544839263292
            # cy = 542.975887548343
            # f = 769.243600037458
            cx = 674.78637996
            cy = 549.15093262
            fx = 770.78529556
            fy = 770.56878243
            intrinsics = np.eye(3)
            intrinsics[0][0] = fx
            intrinsics[1][1] = fy
            intrinsics[0][2] = cx
            intrinsics[1][2] = cy
            intrinsics = intrinsics.astype(np.float32)

            # pose: no quat but matrix
            poses = []
            with open(os.path.join(abs_path, "pose.txt"), "r") as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    # Each line contains a homogenous camera-to-world transformation matrix (flattened in row-major order) corresponding to each frame.
                    line = np.array(line, dtype=np.float32)
                    pose = line.reshape(4, 4, order = "F")
                    poses.append(pose)

            camera_pose = poses[num]

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = self._read_depthmap(depthpath)

            ######## ready to change the
            # if mask_bg:
            #     # load object mask
            #     maskpath = self._get_maskpath(obj, instance, view_idx)
            #     maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
            #     maskmap = (maskmap / 255.0) > 0.1
            #
            #     # update the depthmap with mask
            #     depthmap *= maskmap

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)

            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue

            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                dataset=self.dataset_label,
                label=osp.join(obj, instance),
                instance=osp.split(impath)[1],
            ))
        return views


if __name__ == "__main__":
    from dust3r.datasets.base.base_stereo_view_dataset import view_name
    from dust3r.viz import SceneViz, auto_cam_size
    from dust3r.utils.image import rgb

    dataset = C3VD(split='train', ROOT="/data_new/luxiaoxi/dataset/medical_slam/C3VD_undistorted", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
    # for idx in range(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 5)
        for view_idx in [0, 1]:
            pts3d = views[view_idx]['pts3d']
            valid_mask = views[view_idx]['valid_mask']
            colors = rgb(views[view_idx]['img'])
            viz.add_pointcloud(pts3d, colors, valid_mask)
            viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                           focal=views[view_idx]['camera_intrinsics'][0, 0],
                           color=(idx * 255, (1 - idx) * 255, 0),
                           image=colors,
                           cam_size=cam_size)
        viz.show()
