import cv2
import os
import time
import numpy as np

# ======= 設定 =======
left_id = 0                     # 左攝影機編號
right_id = 2                    # 右攝影機編號
save_dir = "captured_frames"    # 儲存資料夾
capture_interval = 1            # 每幾秒擷取一次
max_frames = 30                 # 最多擷取張數

# 建立左右資料夾
left_dir = os.path.join(save_dir, "left")
right_dir = os.path.join(save_dir, "right")
stero_dir = os.path.join(save_dir, "stereo")
os.makedirs(left_dir, exist_ok=True)
os.makedirs(right_dir, exist_ok=True)
os.makedirs(stero_dir, exist_ok=True)

# 開啟攝影機
cap_left = cv2.VideoCapture(left_id)
cap_right = cv2.VideoCapture(right_id)

if not cap_left.isOpened():
    print(f"無法開啟左攝影機 /dev/video{left_id}")
    exit()
if not cap_right.isOpened():
    print(f"無法開啟右攝影機 /dev/video{right_id}")
    exit()

frame_count = 0
last_capture_time = time.time()

print("開始擷取左右影像，按 q 結束...\n")

while frame_count < max_frames:
    ret_left, frame_left = cap_left.read()
    ret_right, frame_right = cap_right.read()

    if not ret_left or not ret_right:
        print("擷取失敗，跳過...")
        continue

    # 並排顯示左右畫面
    combined = cv2.hconcat([frame_left, frame_right])
    cv2.imshow('Stereo Preview', combined)

    # 是否儲存
    current_time = time.time()
    if current_time - last_capture_time >= capture_interval:
        left_path = os.path.join(left_dir, f"frame_{frame_count:03d}_left.jpg")
        right_path = os.path.join(right_dir, f"frame_{frame_count:03d}_right.jpg")

        cv2.imwrite(left_path, frame_left)
        cv2.imwrite(right_path, frame_right)
        print(f"[✓] 儲存 {left_path} 與 {right_path}")

        stero_path = os.path.join(stero_dir, f"stereo_{frame_count:03d}.jpg")
        canvas = np.hstack((frame_left, frame_right))
        cv2.imwrite(stero_path, canvas)
        print(f"[✓] 儲存 {stero_path}")

        frame_count += 1
        last_capture_time = current_time

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
print("\n✅ 擷取結束。")
