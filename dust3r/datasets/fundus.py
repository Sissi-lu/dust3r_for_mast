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


class Fundus(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        assert self.split is not None
        self.mask_bg = mask_bg
        self.dataset_label = 'fundus'

        datasets = []
        if self.split == "train":
            datasets = ["s1", "s2", "s3",  "s4", "s5"]
        elif self.split == "test":
            datasets = ["s7", "s8", "s9"]
        # remove dataset_4, dataset_5
        # datasets = ["dataset_1"]


        self.pairs = []
        self.scenes = []

        # for dataset in datasets:
        #     for frame_id in sorted(os.listdir(osp.join(ROOT, dataset))):
        #         for img_id in sorted(os.listdir(osp.join(ROOT, dataset, frame_id, "imgL"))):
        #             self.pairs.append([osp.join(ROOT, dataset, frame_id, "imgL", img_id),
        #                                osp.join(ROOT, dataset, frame_id, "imgR", img_id)])


        # for dataset in datasets:
        #     for frame_id in sorted(os.listdir(osp.join(ROOT, dataset))):
        #         # for img_id in sorted(os.listdir(osp.join(ROOT, dataset, frame_id, "imgL"))):
        #         img_id = "00001.png"
        #         self.pairs.append([osp.join(ROOT, dataset, frame_id, "imgL", img_id),
        #                            osp.join(ROOT, dataset, frame_id, "imgR", img_id)])

        ROOT = "/data/luxiaoxi/dataset/medical_depth/Fundus_preprocessed/s1/001"
        imgs = sorted(os.listdir("/data/luxiaoxi/dataset/medical_depth/Fundus_preprocessed/s1/001/imgL"))
        for img in imgs:
            self.pairs.append([osp.join(ROOT, "imgL", img),
                               osp.join(ROOT, "imgR", img)])


        self.scene = sorted(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _read_depthmap(self, depth_path):

        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Loads as uint8 (0-255)

        # # Define clipping planes in meters
        # clip_start = 40  # Near clipping plane (meters)
        # clip_end = 100.0  # Far clipping plane (meters)

        # Convert uint8 depth values (0-255) to metric depth
        depth_map = depth_map/256  # Normalize to 0-1
        # metric_depth = (clip_start + (clip_end - clip_start) * normalized_depth).astype("float32")
        return depth_map.astype("float32")

    def _get_views(self, idx, resolution, rng):
        pairs = self.pairs[idx]
        views = []

        parameter_path = str(pairs[0]).replace("imgL", "calib").replace(".png", ".json")

        with open(parameter_path, "r") as f:
            calib = json.load(f)

        for idx, one_side in enumerate(pairs):
            rgb_path = one_side
            rgb_image = imread_cv2(rgb_path)

            depth_path = str(one_side).replace("img", "depthnew")
            depth_map = self._read_depthmap(depth_path)

            # -------use world matrix generated directly from blender----#
            if idx == 0:
                intrinsics = np.array(calib["k_l"], dtype="float32")
                # camera_pose = np.linalg.inv(np.array(calib["w2cl"], dtype="float32"))
                camera_pose = np.array(calib["w2cl"], dtype='float32')
                # in fact, it's already c2w, with respect to world coordinate frame
            else:
                intrinsics = np.array(calib["k_r"], dtype="float32")
                # camera_pose = np.linalg.inv(np.array(calib["w2cr"], dtype="float32"))
                camera_pose = np.array(calib["w2cr"], dtype='float32')
                # in fact, it's already c2w, with respect to world coordinate frame



            # ------use the quaternion and transform it into rotation matrix ----#
            # if idx == 0:
            #     intrinsics = np.array(calib["k_l"], dtype="float32")
            #     camera_pose = np.array(calib["l2r"], dtype="float32")
            #     camera_pose = np.linalg.inv(camera_pose)
            # else:
            #     intrinsics = np.array(calib["k_r"], dtype="float32")
            #     camera_pose = np.eye(4, dtype="float32")


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

    dataset = Fundus(split='train', ROOT="/data/luxiaoxi/dataset/medical_depth/Fundus_preprocessed", resolution=224, aug_crop=16)

    # for idx in np.random.permutation(len(dataset)):
    for idx in range(len(dataset)):
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

        diff_depth = depth_left - depth_right
        plt.imshow(diff_depth, cmap="grey")
        plt.title("Difference of depth images")
        plt.show()

        diff = rgb_left - rgb_right
        plt.imshow(diff, cmap="grey")
        plt.title("Difference of RGB images")
        plt.show()



        print(view_name(views[0]), view_name(views[1]))
        viz = SceneViz()
        poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
        cam_size = max(auto_cam_size(poses), 1)
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
