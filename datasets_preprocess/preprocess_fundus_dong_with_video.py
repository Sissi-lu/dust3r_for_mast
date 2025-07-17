import cv2
import os
from tqdm import tqdm
import numpy as np
import imageio

root_dir = "/data/luxiaoxi/dataset/medical_depth/final_version_processed"
scene_folders = [f for f in os.listdir(root_dir) if not "split" in f]
save_dir = "/data/luxiaoxi/dataset/medical_depth/final_version_processed_videos"
os.makedirs(save_dir, exist_ok=True)

fps = 5  # 保存视频的帧率
size = (512, 512)  # 保存视频的大小

for scene_folder_name in scene_folders:
    video_save_path = os.path.join(save_dir, scene_folder_name)
    os.makedirs(video_save_path, exist_ok=True)
    for i in tqdm(range(1, 129)): # subscene_id
        videoWriter = cv2.VideoWriter(os.path.join(video_save_path, '%03d.avi'%i), cv2.VideoWriter.fourcc(*'XVID'), fps, size)
        img_list = [f for f in os.listdir(os.path.join(root_dir, scene_folder_name, "left", "imgs")) if f.startswith("%03d"%i)]
        img_list = sorted(img_list)
        for img_ in img_list:
            img_path = os.path.join(root_dir, scene_folder_name, "left", "imgs", img_)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            videoWriter.write(img)