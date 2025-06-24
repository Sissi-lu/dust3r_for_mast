import shutil
import os
import glob

datadir = "/data_new/luxiaoxi/dataset/medical_slam/C3VD_undistorted/cecum_t1_a_under_review/images"
imgs_name = sorted(glob.glob(os.path.join(datadir, "*_color.png")), key=lambda x:int(x.split('/')[-1].split('_')[0]))

for idx, img_name in enumerate(imgs_name):
    num = int(img_name.split('/')[-1].split('_')[0])
    old_name = img_name.split('/')[-1]
    new_name = "%04d_color.png"%num
    new_img_name = img_name.replace(old_name, new_name)
    os.rename(img_name, new_img_name)