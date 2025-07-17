import os
import numpy as np
from tqdm import tqdm

data_root = "/data/luxiaoxi/dataset/medical_depth/final_version_processed"
# qualified_set = {
#    "s5": [66, 51, 65, 73, 78],
# #     "  "s1": [60, 67, 71, 74, 92, 108, 118, 119, 120, 57, 70],
#     "s2": [18, 37, 42, 92, 103, 124, 119, 23, 65, 94, 122],
#     "s3": [13, 20, 51, 61, 66],
#     "s4": [61, 69, 72, 116, 117, 120, 6, 32, 45, 54, 51, 65],
#    s6": [23, 26, 62, 64, 116, 125, 22, 74, 89, 100],
#     "s7": [21, 27, 39, 74, 81, 83, 97, 109, 19, 34, 47, 53, 67, 102, 110, 124]
# }

qualified_set = {
    "s1": [60, 67, 71, 74, 108, 57, 70],
    "s2": [65],
    "s3": [51, 61, 66],
    "s4": [61, 69, 72, 116, 117, 120, 6, 32, 45, 54, 51, 65],
    "s5": [66, 51, 65],
    "s6": [23, 26, 62, 64, 116, 125, 22, 74, 89, 100],
    "s7": [21, 27, 39, 74, 81, 83, 97, 109, 19, 34, 47, 53, 67, 102, 110, 124]
}

train_file = os.path.join(data_root, "train.txt")
test_file = os.path.join(data_root, "test.txt")

scene_folder = [f for f in os.listdir(data_root) if f != "split" and not f.endswith("txt")]
for scene in scene_folder:
    scene_id = scene.split("_")[0]
    if scene_id == "s3" or scene_id == "s5":
        qualified_scene = qualified_set[scene_id]
        all_imgs = os.listdir(os.path.join(data_root, scene, "left", "imgs"))
        for img in tqdm(all_imgs):
            if int(img[:3]) in qualified_scene:
                if os.path.exists(os.path.join(data_root, scene, "right", "imgs", img)):
                    with open(test_file, "a") as f:
                        f.write(os.path.join(scene, "left", "imgs", img)+"\n")
    else:
        qualified_scene = qualified_set[scene_id]
        all_imgs = os.listdir(os.path.join(data_root, scene, "left", "imgs"))
        for img in tqdm(all_imgs):
            if int(img[:3]) in qualified_scene:
                if os.path.exists(os.path.join(data_root, scene, "right", "imgs", img)):
                    with open(train_file, "a") as f:
                        f.write(os.path.join(scene, "left", "imgs", img) + "\n")
