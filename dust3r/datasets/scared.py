# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# See datasets_preprocess/preprocess_co3d.py
# --------------------------------------------------------
import os.path as osp
import json
import itertools
from collections import deque

import cv2
import numpy as np


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class SCARED(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'SCARED'

        # load all scenes
        if self.split == "train":
            with open(osp.join(self.ROOT, f'train.json'), 'r') as f:
                self.scenes = json.load(f)
        elif self.split == "test":
            with open(osp.join(self.ROOT, f'val.json'), 'r') as f:
                self.scenes = json.load(f)
        self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
        self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                       for k2, v2 in v.items()}
        self.scene_list = list(self.scenes.keys())

        # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
        # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
        self.combinations = [(i, j)
                             for i, j in itertools.combinations(range(100), 2)
                             if 0 < abs(i - j) <= 30 and abs(i - j) % 5 == 0]

        self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, instance, 'image_02/data/frame_data', f'frame_data%06d.json'%view_idx)

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, instance, 'image_02/data', f'%010d.png'%view_idx)

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, instance, 'image_02/data/groundtruth', f'scene_points%06d.tiff'%view_idx)

    # def _get_maskpath(self, obj, instance, view_idx):
    #     return osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')

    def _read_depthmap(self, depthpath, get):
        depthmap = cv2.imread(depthpath, 3)
        depth_gt = depthmap[:, :, 0]
        gt_depth = depth_gt[0:1024, :].astype(np.float32)
        return gt_depth

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

            view_idx = image_pool[im_idx]

            impath = self._get_impath(obj, instance, view_idx)
            depthpath = self._get_depthpath(obj, instance, view_idx)

            # load camera params
            metadata_path = self._get_metadatapath(obj, instance, view_idx)

            with open(metadata_path, "r") as f:
                input_metadata = json.load(f)

            camera_pose = np.array(input_metadata['camera-pose'], dtype=np.float32)
            intrinsics = np.array(input_metadata['camera-calibration']['KL'], dtype=np.float32)
            # distortion = np.array(input_metadata['camera_intrinsics']['DL'], dtype=np.float32)


            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = self._read_depthmap(depthpath, input_metadata)

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

    dataset = SCARED(split='test', ROOT="/data_new/luxiaoxi/dataset/medical_depth/SCARED", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 0.001)
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
