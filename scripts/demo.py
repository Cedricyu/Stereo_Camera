import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import load_stereo_parameters
from compute_disparity import compute_disparity, compute_disparity_libsgm
from compute_cloud import compute_pointcloud, save_pointcloud_ply, visualize_pointcloud
import os
import time

save_dir = "../outputs/video"
os.makedirs(save_dir, exist_ok=True)

# 1. 讀取 stereo_camera.yaml
stereo_yaml_path = "../configs/stereo_camera.yaml"
mtxL, distL, mtxR, distR, R, T = load_stereo_parameters(stereo_yaml_path)

image_size = (640, 480)

# Open camera devices (modify indices if needed)
capL = cv2.VideoCapture(0)  # /dev/video0
capR = cv2.VideoCapture(2)

# Set resolution
capL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Read one frame to get size
retL, frameL = capL.read()
retR, frameR = capR.read()
h, w = frameL.shape[:2]
image_size = (w, h)

# Stereo rectify maps
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtxL, distL, mtxR, distR, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)

left_map1, left_map2 = cv2.initUndistortRectifyMap(
    mtxL, distL, R1, P1, image_size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    mtxR, distR, R2, P2, image_size, cv2.CV_16SC2)

mtxL = P1
mtxR = P2

recording = False
video_writer = None
record_start_time = None
record_duration = 5
fps_out = 20.0
clouds = []
idx = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')

def start_recording():
    global recording, record_start_time, frame_count
    print(f"[INFO] Start recording frames...")
    recording = True
    record_start_time = time.time()
    frame_count = 0

def stop_recording():
    global recording, idx
    print(f"[✓] Finished recording {frame_count} frames to {save_dir}")
    idx += 1
    recording = False

left_dir = os.path.join(save_dir, "left")
point_dir = os.path.join(save_dir, "depth_npy")
depth_dir = os.path.join(save_dir, "depth")
os.makedirs(left_dir, exist_ok=True)
os.makedirs(point_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

time_t = 0
while True:
    retL, frameL = capL.read()
    retR, frameR = capR.read()
    if not retL or frameR is None:
        print("Error reading from cameras")
        break

    grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

    rectL = cv2.remap(grayL, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
    rectR = cv2.remap(grayR, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)

    rectL_color = cv2.remap(frameL, left_map1, left_map2, interpolation=cv2.INTER_LINEAR)
    rectR_color = cv2.remap(frameR, right_map1, right_map2, interpolation=cv2.INTER_LINEAR)
    
    if recording:
        start = time.time()

    disparity = compute_disparity_libsgm(rectL, rectR)

    if recording:
        end = time.time()
        time_t += (end - start)

    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_bgr = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((rectL_color, rectR_color, disp_bgr))

    for y in range(0, combined.shape[0], 40):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    cv2.imshow("Stereo Rectified + Disparity", combined)

    if recording:
        frame_name = f"{frame_count:04d}"

        # 存左相機彩色校正圖
        frame_path = os.path.join(left_dir, f"{frame_name}.png")
        cv2.imwrite(frame_path, rectL_color)
        print(f"[✓] Saved left image to {frame_path}")

        disp_img_path = os.path.join(depth_dir, f"disp_{frame_name}.png")
        cv2.imwrite(disp_img_path, disp_vis)
        print(f"[✓] Saved disparity image to {disp_img_path}")
        
        # 點雲
        cloud_path = os.path.join(point_dir, f"disp_{frame_name}.npy")
        np.save(cloud_path, disparity)

        print(f"[✓] Saved frame {frame_name} to left/ and depth_npy/")

        frame_count += 1

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        if not recording:
            start_recording()
        else:
            stop_recording()
            fps = frame_count / time_t if time_t > 0 else 0.0
            print(f"[INFO] Frame {frame_count} processed in {time_t:.3f} seconds ({fps:.2f} FPS)")

    elif key == 27:  # ESC
        if recording:
            stop_recording()
        break


capL.release()
capR.release()
cv2.destroyAllWindows()
