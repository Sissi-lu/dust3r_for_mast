import os.path as osp
import json
import itertools
from collections import deque
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append("/data/luxiaoxi/code_proj/depth_estimation/MedicalMast3R/dust3r")

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2

# from utils.image import imread_cv2
# from base.base_stereo_view_dataset import BaseStereoViewDataset

def extract_K(data):
    fx = data['lens']*512/data['sensor_width']
    fy = data['lens']*512/data['sensor_height']
    intrinsic = np.eye(3, dtype="float32")
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = 256
    intrinsic[1, 2] = 256
    return intrinsic

def quaternion2rotation_matrix(q):
    R = np.eye(3, dtype="float32")
    q = q / np.sqrt(np.dot(q, q))  # Normalize the quaternion

    w, x, y, z = q  # Unpack quaternion components

    # Compute the rotation matrix elements
    r11 = 1 - 2 * y * y - 2 * z * z
    r12 = 2 * x * y - 2 * w * z
    r13 = 2 * x * z + 2 * w * y

    r21 = 2 * x * y + 2 * w * z
    r22 = 1 - 2 * x * x - 2 * z * z
    r23 = 2 * y * z - 2 * w * x

    r31 = 2 * x * z - 2 * w * y
    r32 = 2 * y * z + 2 * w * x
    r33 = 1 - 2 * x * x - 2 * y * y

    # Construct the 3x3 matrix
    rotation_matrix = np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])

    return rotation_matrix

def change_location_dict2array(location):
    pos = [location['x'], location['y'], location['z']]
    return np.array(pos)

def extract_calib(data):
    # left camera
    intrinsic_left = extract_K(data['camera_l'])
    intrinsic_right = extract_K(data['camera_r'])
    w2cl = np.array(data['camera_l']['word_matrix'])
    w2cr = np.array(data['camera_r']['word_matrix'])

    quaternion_left = np.array(data['camera_l']['quaternion_rotation'])
    quaternion_right = np.array(data['camera_r']['quaternion_rotation'])

    pos_left = change_location_dict2array(data['camera_l']['location'])
    pos_right = change_location_dict2array(data['camera_r']['location'])
    return intrinsic_left, w2cl, intrinsic_right, w2cr, quaternion_left, quaternion_right, pos_left, pos_right

class FundusDong(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, running_list, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        assert self.split is not None
        self.mask_bg = mask_bg
        self.dataset_label = 'fundus'


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

        # new_running_list = [f for f in running_list if f.split('/')[0].split('_')[0] == "s4"]
        # running_list = sorted(running_list, key=lambda x: (x.split('/')[-1].split('.')[0][:3], x.split('/')[-1].split('.')[0][3:]))

        scene_folder = sorted(f for f in os.listdir(self.ROOT) if not "split" in f and not f.endswith("txt"))
        new_running_list = []
        for scene in scene_folder:
            for i in range(1, 129):
                img_path = os.path.join(self.ROOT, scene, "left", "imgs", "%03d26.png"%i)
                new_running_list.append(img_path)

        # for img_path in sorted(running_list):
        for img_path in sorted(new_running_list):
            img_path = osp.join(self.ROOT, img_path)
            if img_path is not None:
                self.pairs.append([img_path,
                                   img_path.replace('left', 'right')])


        self.scene = sorted(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _read_depthmap(self, depth_path):

        # depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Loads as uint8 (0-255)
        depth_map = np.load(depth_path)
        # # Define clipping planes in meters
        # clip_start = 40  # Near clipping plane (meters)
        # clip_end = 100.0  # Far clipping plane (meters)

        # Convert uint8 depth values (0-255) to metric depth
        # depth_map = depth_map/256  # Normalize to 0-1
        # depth_map = depth_map
        # metric_depth = (clip_start + (clip_end - clip_start) * normalized_depth).astype("float32")
        return depth_map.astype("float32")

    def quat_to_matrix(self, pos, quat):
        # there is no difference between T0 and T1
        x, y, z = pos
        qw, qx, qy, qz = quat
        rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = [x, y, z]
        return T

    def _get_views(self, idx, resolution, rng):
        pairs = self.pairs[idx]
        views = []
        scene = pairs[0].split('/')[6]
        scene_number = int(pairs[0].split('/')[-1].split('.')[0][:3])
        picture_number = int(pairs[0].split('/')[-1].split('.')[0][3:])
        # print("Series: %s, Scene number: %d"%(scene, scene_number))

        intrinsic_number = (scene_number - 1)*50-1 + picture_number
        # parameter_path = str(pairs[0]).replace("imgL", "calib").replace(".png", ".json")
        parameter_path = osp.join(self.ROOT, scene, 'instrincs.json')
        with open(parameter_path, "r") as f:
            calib = json.load(f)

        assert calib[intrinsic_number]['name'] == pairs[0].split('/')[-1].split(',')[0], "calib name is %s and pair name is %s"%(calib[intrinsic_number]['name'], pairs[0].split('/')[-1].split(',')[0])
        calib_org =  calib[intrinsic_number]

        intrinsic_left, w2cl, intrinsic_right, w2cr, ql, qr, posl, posr = extract_calib(calib_org)

        camera_l = calib_org['camera_l']
        camera_r = calib_org['camera_r']

        t_l = [camera_l['location']['x'], camera_l['location']['y'], camera_l['location']['z']]
        t_r = [camera_r['location']['x'], camera_r['location']['y'], camera_r['location']['z']]

        q_l = camera_l['quaternion_rotation']
        q_r = camera_r['quaternion_rotation']

        T_l = self.quat_to_matrix(t_l, q_l).astype("float32")
        T_r = self.quat_to_matrix(t_r, q_r).astype("float32")

        # if t_l == t_r:
        #     T_l[:3, 3] = w2cl[:3, 3]
        #     T_r[:3, 3] = w2cr[:3, 3]
        # validation the effectiveness of the quaternion and w2c
        # t = posl - posr
        # wrong pos

        # w_posl = w2cl[:3, 3]
        # w_posr = w2cr[:3, 3]


        R_ql = quaternion2rotation_matrix(ql)
        R_qr = quaternion2rotation_matrix(qr)

        # Rot_l2r = np.eye(4, dtype="float32")
        # Rot_l2r[:3, :3] = R_ql
        # Rot_l2r[:3, 3] = t

        new_calib = dict()
        new_calib["k_l"] = intrinsic_left.tolist()
        new_calib["k_r"] = intrinsic_right.tolist()
        new_calib["w2cl"] = w2cl.tolist()
        new_calib["w2cr"] = w2cr.tolist()

        # new_calib["l2r"] = Rot_l2r.tolist()


        for idx, one_side in enumerate(pairs):
            rgb_path = one_side
            rgb_image = imread_cv2(rgb_path)

            depth_path = str(one_side).replace("imgs", "metric_depth").replace("png", "npy")
            depth_map = self._read_depthmap(depth_path)

            # -------use world matrix generated directly from blender----#
            if idx == 0:
                intrinsics = np.array(new_calib["k_l"], dtype="float32")
                # camera_pose = np.linalg.inv(np.array(calib["w2cl"], dtype="float32"))
                camera_pose = np.array(new_calib["w2cl"], dtype='float32')
                # camera_pose = np.linalg.inv(camera_pose)
                # camera_pose = np.eye(4, dtype="float32")
                # in fact, it's already c2w, with respect to world coordinate frame
                # camera_pose = T_l
                # camera_pose = np.linalg.inv(camera_pose)
            else:
                intrinsics = np.array(new_calib["k_r"], dtype="float32")
                # camera_pose = np.linalg.inv(np.array(calib["w2cr"], dtype="float32"))
                camera_pose = np.array(new_calib["w2cr"], dtype='float32')
                # camera_pose = np.linalg.inv(camera_pose)
                # camera_pose = np.array(new_calib["l2r"], dtype="float32")
                # camera_pose = np.linalg.inv(camera_pose)
                # camera_pose = T_r
                # camera_pose = np.linalg.inv(camera_pose)

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

    data_root = "/data/luxiaoxi/dataset/medical_depth/final_version_processed"
    train_running_list = []
    with open(os.path.join(data_root, "new_split", "train.txt")) as file:
        for line in file.readlines():
            # line = line.strip('\n').split(',')[0]
            line = line.strip('\n')
            train_running_list.append(line)

    dataset = FundusDong(split='train', ROOT=data_root, running_list=train_running_list, resolution=512, aug_crop=16)

    # for idx in np.random.permutation(len(dataset)):
    for idx in range(len(dataset)):
        views = dataset[idx]
        assert len(views) == 2
        # depth_left = views[0]["depthmap"]
        # depth_right = views[1]["depthmap"]
        # rgb_left = views[0]["img"].permute(1, 2, 0)
        # rgb_right = views[1]["img"].permute(1, 2, 0)
        #
        # depth = np.concatenate([depth_left, depth_right], axis=1)
        # plt.imshow(depth, cmap="jet")
        # plt.show()
        #
        # rgb_combine = torch.concat([rgb_left, rgb_right], dim=1)
        # plt.imshow(rgb_combine)
        # plt.show()
        #
        # diff_depth = depth_left - depth_right
        # plt.imshow(diff_depth, cmap="grey")
        # plt.title("Difference of depth images")
        # plt.show()
        #
        # diff = rgb_left - rgb_right
        # plt.imshow(diff, cmap="grey")
        # plt.title("Difference of RGB images")
        # plt.show()



        # print(view_name(views[0]), view_name(views[1]))

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
