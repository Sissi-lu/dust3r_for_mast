import os
import shutil

new_data_dir = "/data_new/luxiaoxi/dataset/medical_slam/C3VD_undistorted"
old_data_dir = "/data_new/luxiaoxi/dataset/medical_slam/C3VD"

folders = [f for f in os.listdir(new_data_dir) if not f.endswith('.txt') and f != 'desc_t4_a_p2_under_review' and f != 'desc_t4_a_p1_under_review']
for folder in folders:
    folder_name = folder.split('_under')[0]
    old_pose_path = os.path.join(old_data_dir, folder_name, 'pose.txt')
    assert os.path.exists(old_pose_path)
    new_pose_path = os.path.join(new_data_dir, folder, 'pose.txt')
    shutil.copyfile(old_pose_path, new_pose_path)