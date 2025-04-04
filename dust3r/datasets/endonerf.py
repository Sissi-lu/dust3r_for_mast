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
import imageio
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2


class Endonerf(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        assert self.split is not None
        self.mask_bg = mask_bg
        self.dataset_label = 'endonerf'

        datasets = []
        if self.split == "train":
            datasets = ["pulling_soft_tissues"]
        elif self.split == "test":
            datasets = ["cutting_tissues_twice"]
        # remove dataset_4, dataset_5
        # datasets = ["dataset_1"]


        self.pairs = []
        self.scenes = []
        for dataset in datasets:
                for img_id in sorted(os.listdir(osp.join(ROOT, dataset, "left"))):
                    self.pairs.append([osp.join(ROOT, dataset, "left", img_id),
                                       osp.join(ROOT, dataset, "right", img_id)])
        self.scene = sorted(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _read_depthmap(self, depth_path):

        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        depth_map = (depth_map.astype(np.float32) / 255)
        return depth_map

    def _get_views(self, idx, resolution, rng):
        pairs = self.pairs[idx]
        views = []


        parameter_path = str(pairs[0]).split("left")[0] + "poses_bounds.npy"
        calib = np.load(parameter_path)

        poses_arr = calib[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = calib[:, -2:].transpose([1, 0])

        sh = imageio.imread(pairs[0]).shape
        poses_arr[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
        poses_arr[2, 4, :] = poses_arr[2, 4, :] * 1.

        for idx, one_side in enumerate(pairs):
            name_id = int(one_side.split("/")[-1].split(".")[0])
            calib_id = calib[name_id, :]
            poses = calib_id[:15].reshape(3, 5)
            rgb_path = one_side
            rgb_image = imread_cv2(rgb_path)

            depth_path = str(one_side).replace("left", "gt_disparity")
            depth_map = self._read_depthmap(depth_path)

            # camera pose is w2c
            if idx == 0:
                intrinsics = np.eye(3, dtype="float32")
                intrinsics[0, 0] = poses[2, 4]
                intrinsics[1, 1] = poses[2, 4]
                intrinsics[0, 2] = poses[1, 4]/2
                intrinsics[1, 2] = poses[0, 4]/2
                camera_pose = np.eye(4, dtype="float32")
                camera_pose[:3, :3] =poses[:3, :3]
            else:
                intrinsics = np.eye(3, dtype="float32")
                intrinsics[0, 0] = poses[2, 4]
                intrinsics[1, 1] = poses[2, 4]
                intrinsics[0, 2] = poses[1, 4] / 2
                intrinsics[1, 2] = poses[0, 4] / 2
                camera_pose = np.eye(4, dtype="float32")
                camera_pose[:3, :3] = poses[:3, :3]

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

    dataset = Endonerf(split='train', ROOT="/data/luxiaoxi/dataset/medical_depth/endonerf", resolution=224, aug_crop=16)

    for idx in np.random.permutation(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        # cam_size = max(auto_cam_size(poses), 0.001)
        cam_size = max(auto_cam_size(poses), 0.1)
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
