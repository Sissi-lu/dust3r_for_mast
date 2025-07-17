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

def extract_K(data):
    fx = data['lens']*512/data['sensor_width']
    fy = data['lens']*512/data['sensor_height']
    intrinsic = np.eye(3, dtype="float32")
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = 256
    intrinsic[1, 2] = 256
    return intrinsic

class FundusSLAM(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'FundusSLAM'

        # load all scenes: set 1-5 as train, set 6-7 as test
        if self.split == "train":
            scenes = ["s%d_processed"%i for i in range(1, 6)]
            scene_ids = range(1, 129)
        elif self.split == "test":
            scenes = ["s%d_processed" % i for i in range(6, 8)]
            scene_ids = range(1, 129)

        self.scenes = dict()
        # for scene in scenes:
        #     for ids in scene_ids:
        #         self.scenes[(scene, "%03d"%ids )] = sorted(glob.glob(os.path.join(self.ROOT, scene, "left", "imgs", "%03d*"%ids)))
        self.scenes = {(scene, "%03d"%ids): sorted(glob.glob(os.path.join(self.ROOT, scene, "left", "imgs", "%03d*.png"%ids))) for scene in scenes for ids in scene_ids}

        self.scene_list = list(self.scenes.keys())

        # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
        # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
        # self.combinations = [(i, j)
        #                      for i, j in itertools.combinations(range(50), 2)
        #                      if 0 < abs(i - j) <= 50 and abs(i - j) % 10 == 0]
        # self.combinations = [(i, j)
        #                      for i, j in itertools.combinations(range(10), 2)
        # #                      if 0 < abs(i - j) <= 10 and abs(i - j) % 2 == 0]
        self.combinations = [(i, i + k)
                             for i in range(50)
                             for k in [1, 2, 3]
                             if i + k < 50]

        self.invalidate = {scene: {} for scene in self.scene_list}

        # self.min_depth = 0.001
        # self.max_depth = 20

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

    def _read_depthmap(self, depthpath):
        # depthmap = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = np.load(depthpath)
        depthmap = depthmap.astype(np.float32)
        # depthmap[depthmap < 0] = 0
        # depthmap[depthmap > self.max_depth] = self.max_depth
        return depthmap

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

            depthpath = impath.replace("imgs", "metric_depth").replace("png", "npy")
            abs_path = os.path.abspath(os.path.join(impath, "../.."))

            # load camera params
            scene = impath.split('/')[6]
            scene_number = int(impath.split('/')[-1].split('.')[0][:3])
            picture_number = int(impath.split('/')[-1].split('.')[0][3:])
            intrinsic_number = (scene_number - 1) * 50 - 1 + picture_number
            # parameter_path = str(pairs[0]).replace("imgL", "calib").replace(".png", ".json")
            parameter_path = osp.join(self.ROOT, scene, 'instrincs.json')
            with open(parameter_path, "r") as f:
                calib = json.load(f)

            assert calib[intrinsic_number]['name'] == impath.split('/')[-1].split(',')[
                0], "calib name is %s and pair name is %s" % (
            calib[intrinsic_number]['name'], impath.split('/')[-1].split(',')[0])
            calib_org = calib[intrinsic_number]
            intrinsics = extract_K(calib_org['camera_l']).astype(np.float32)
            camera_pose = np.array(calib_org['camera_l']['word_matrix']).astype(np.float32)


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

    dataset = FundusSLAM(split='test', ROOT="/data_new/luxiaoxi/dataset/medical_depth/final_version_processed", resolution=512, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
    # for idx in range(len(dataset)):
    # for idx in [457]:
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 1)
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
