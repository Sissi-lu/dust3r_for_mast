import os

# root_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/"
# folder_list = [f for f in os.listdir(root_dir) if f != "split"]
# k = 0
# for folder in folder_list:
#     for one_side in ["left", "right"]:
#         imgs_path = os.path.join(root_dir, folder, one_side, "imgs")
#         mds_path = os.path.join(root_dir, folder, one_side, "metric_depth")
#         for img in os.listdir(imgs_path):
#             img_path = os.path.join(imgs_path, img)
#             depth_path = os.path.join(mds_path, img.replace("png", "npy"))
#             if not os.path.exists(depth_path):
#                 print(depth_path)
#                 k=k+1
#                 os.remove(img_path)
#
# print(k)
import os

remove_list = []
# with open('./remove_img_files.txt', 'r') as f:
#     for line in f.readlines():
#         line = line.strip().split('/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/')[1]
#         remove_list.append(line)

# new_split = os.path.join("/data/luxiaoxi/dataset/medical_depth/final_version_processed", "new_split")
# os.makedirs(new_split, exist_ok=True)

split_dir = "/data/luxiaoxi/dataset/medical_depth/final_version_processed/split"
new_list = []
# k=0
for file in ["train", "test", "val"]:
    k=0
    with open(os.path.join(split_dir, "%s.txt"%file), 'r') as f:
        for line in f.readlines():
            newline = line.strip().split(',')[0]
            k = k+1
            # for remove in remove_list:
            #     if  newline== remove.replace('metric_depth', "imgs").replace("npy", "png").replace("right", "left"):
            #         print(newline)
            #         break
            #     k = k+1
            # if k == len(remove_list):
            #     new_list.append(line)

        print("%s has %d lines" % (file, k))
    # with open(os.path.join(new_split, "%s.txt"%file), 'a') as f:
    #     for line in new_list:
    #         f.write("%s\n"%line)