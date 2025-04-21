import os

root_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/"
folder_list = [f for f in os.listdir(root_dir) if f != "split"]
k = 0
for folder in folder_list:
    for one_side in ["left", "right"]:
        imgs_path = os.path.join(root_dir, folder, one_side, "imgs")
        mds_path = os.path.join(root_dir, folder, one_side, "metric_depth")
        for img in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, img)
            depth_path = os.path.join(mds_path, img.replace("png", "npy"))
            if not os.path.exists(depth_path):
                print(depth_path)
                k=k+1
                os.remove(img_path)

print(k)