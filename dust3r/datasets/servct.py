# --------------------------------------------------------
# Dataloader for preprocessed SERVCT dataset
# See datasets_preprocess/preprocess_SERVCT.py
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


class Servct(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, running_list, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        assert self.split is not None
        self.mask_bg = mask_bg
        self.dataset_label = 'Servct'

        self.pairs = []

        for img_name in sorted(running_list):
            if img_name is not None:
                self.pairs.append([osp.join(ROOT, "left", img_name),
                                   osp.join(ROOT, "right", img_name)])
        self.scene = sorted(self.pairs)
        print(len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def _read_depthmap(self, depth_path):

        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_map = (depth_map.astype(float) / 256).astype("float32")
        return depth_map

    def _get_views(self, idx, resolution, rng):
        pairs = self.pairs[idx]
        views = []


        parameter_path = str(pairs[0]).replace("left", "Rectified_calibration").replace(".png", ".json")
        with open(parameter_path, "r") as f:
            calib = json.load(f)

        for idx, one_side in enumerate(pairs):
            rgb_path = one_side
            rgb_image = imread_cv2(rgb_path)
            if idx == 0:
                depth_path = str(one_side).replace("left", "depthL")
            else:
                depth_path = str(one_side).replace("right", "depthR")
            depth_map = self._read_depthmap(depth_path)

            P1 = np.array(calib['P1']['data']).reshape(3, 4)
            P2 = np.array(calib['P2']['data']).reshape(3, 4)
            Q = np.array(calib['Q']['data']).reshape(4, 4)
            Tx = P2[0, 3]/P2[0, 0]

            if idx == 0:
                intrinsics_list = calib['P1']['data']

                intrinsics = np.zeros([3, 3], dtype="float32")
                intrinsics[0, :] = intrinsics_list[0:3]
                intrinsics[1, :] = intrinsics_list[4:7]
                intrinsics[2, :] = intrinsics_list[8:11]
                camera_pose = np.eye(4, dtype="float32")
            else:
                intrinsics_list = calib['P2']['data']
                intrinsics = np.zeros([3, 3], dtype=np.float32)
                intrinsics[0, :] = intrinsics_list[0:3]
                intrinsics[1, :] = intrinsics_list[4:7]
                intrinsics[2, :] = intrinsics_list[8:11]

                extrinsic = np.eye(4, dtype="float32")
                extrinsic[0, 3] = intrinsics_list[3]/intrinsics_list[0]

                # extrinsic is respect to camera coordinate frame
                camera_pose = np.linalg.inv(np.array(extrinsic, dtype="float32"))

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

    dataset = Servct(split='train', ROOT="/data/luxiaoxi/dataset/medical_depth/SERV-CT_preprocessed", resolution=224, aug_crop=16)

    # for idx in np.random.permutation(len(dataset)):
    for idx in tqdm(range(len(dataset))):
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