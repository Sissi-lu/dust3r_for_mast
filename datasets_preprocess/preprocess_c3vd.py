# Script to calibrate using C3VD checkerboard calibration frames and undistort registered
# videos in the C3VD dataset.
#
# Author: Akshay Paruchuri (akshay@cs.unc.edu)
import cv2
import numpy as np
import os
from tqdm import tqdm

def main():
    # Set your chessboard dimensions (number of inner corners)
    chessboard_size = (10, 15)  # Change this to match your calibration target

    # Read the directory containing TIFF frames
    calib_images_folder = "/data_new/luxiaoxi/dataset/medical_slam/C3VD/calibration_frames"  # Update this
    registered_videos_path = "/data_new/luxiaoxi/dataset/medical_slam/C3VD"

    # Create arrays to store object points and image points from all the images
    obj_points = []  # 3D points in real-world space
    img_points = []  # 2D points in the image plane

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    # Create folders to organize undistorted images
    undistorted_folder = "/data_new/luxiaoxi/dataset/medical_slam/C3VD_undistorted"
    os.makedirs(undistorted_folder, exist_ok=True)

    calib_images = [f for f in os.listdir(calib_images_folder)]

    for image_name in calib_images:
        image_path = os.path.join(calib_images_folder, image_name)
        image = cv2.imread(image_path, -1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Save the calibration parameters to a file under the main undistorted folder
    with open(os.path.join(undistorted_folder, "output_calibration.txt"), "w") as file:
        fx = mtx[0, 0]
        fy = mtx[1, 1]
        cx = mtx[0, 2]
        cy = mtx[1, 2]
        file.write(f"Pinhole {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f} 0")

    registered_videos = [f for f in os.listdir(registered_videos_path) if f != "calibration_frames" and f != "split_files" and not f.endswith('.zip')]

    for registered_video in registered_videos:
        folder_path = os.path.join(registered_videos_path, registered_video)
        output_folder_path = os.path.join(undistorted_folder, registered_video+"_under_review")

        color_folder = os.path.join(output_folder_path, "images")
        depth_folder = os.path.join(output_folder_path, "depths")
        normals_folder = os.path.join(output_folder_path, "normals")
        occlusion_folder = os.path.join(output_folder_path, "occlusion")
        optical_flow_folder = os.path.join(output_folder_path, "optical_flow")

        # Create the undistorted folders if they don't exist
        os.makedirs(color_folder, exist_ok=True)
        os.makedirs(depth_folder, exist_ok=True)
        os.makedirs(normals_folder, exist_ok=True)
        os.makedirs(occlusion_folder, exist_ok=True)
        os.makedirs(optical_flow_folder, exist_ok=True)

        seq_images = [f for f in os.listdir(folder_path)]

        # Undistort and organize the images
        for image_name in tqdm(seq_images):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path, -1)
            image_type = None

            if image_name.endswith("_color.png"):
                undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
                image_type = "images"
            elif image_name.endswith("_depth.tiff"):
                undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
                image_type = "depths"
            elif image_name.endswith("_normals.tiff"):
                undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
                image_type = "normals"
            elif image_name.endswith("_occlusion.png"):
                undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
                image_type = "occlusion"
            if image_name.endswith("_flow.tiff"):
                undistorted_image = cv2.undistort(image, mtx, dist, None, mtx)
                image_type = "optical_flow"

            if image_type:
                image_output_folder = os.path.join(output_folder_path, image_type)
                image_output_path = os.path.join(image_output_folder, image_name)
                cv2.imwrite(image_output_path, undistorted_image)
                print(f"Undistorted and saved {image_type} image: {image_name} to {image_output_path}")

if __name__ == "__main__":
    main()