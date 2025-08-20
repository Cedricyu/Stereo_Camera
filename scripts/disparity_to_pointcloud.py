import cv2
import numpy as np
import argparse

def disparity_to_pointcloud(disp, left_bgr, fx, fy, cx, cy, baseline_m,
                            max_depth=5, out_ply="cloud.ply"):
    """
    disp: disparity map (float32, already /16 if from OpenCV SGBM)
    left_bgr: 左影像 (H,W,3)
    fx, fy, cx, cy: 左相機內參
    baseline_m: 基線 (公尺)
    """
    h, w = disp.shape
    Q = np.float32([[1, 0, 0, -cx],
                    [0, 1, 0, -cy],
                    [0, 0, 0, fx],
                    [0, 0, 1.0/baseline_m, 0]])

    # 轉成 3D 座標
    points_3D = cv2.reprojectImageTo3D(disp, Q)
    mask = disp > 0.2
    mask &= (points_3D[...,2] < max_depth)

    pts = points_3D[mask]
    cols = left_bgr[mask]

    # === 印數值範圍 ===
    print("Disparity valid range:", np.min(disp[mask]), np.max(disp[mask]))
    print("X range:", np.min(pts[:,0]), np.max(pts[:,0]))
    print("Y range:", np.min(pts[:,1]), np.max(pts[:,1]))
    print("Z range:", np.min(pts[:,2]), np.max(pts[:,2]))

    pts[:, 0] *= -1  # 左右鏡像
    pts[:, 1] *= -1  # 上下翻轉



    save_ply(out_ply, pts, cols)

def save_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(verts))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x,y,z), (r,g,b) in zip(verts, colors):
            f.write(f"{x} {y} {z} {r} {g} {b}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute disparity with StereoSGBM")
    ap.add_argument("--left", required=True, help="左影像路徑（rectified, grayscale 或彩色）")
    ap.add_argument("--disp", required=True, help="視差圖")
    ap.add_argument("--max_depth", type=float, required=False, help="視差圖")
    args = ap.parse_args()

    fx, fy, cx, cy = 537.8920, 539.1635, 630.3863, 327.5218
    baseline_m = 0.117  # 11.7 cm
    # disparity = np.load(args.disp).astype(np.float32)
    disparity = cv2.imread(args.disp, cv2.IMREAD_GRAYSCALE)

    left_bgr = cv2.imread(args.left, cv2.IMREAD_UNCHANGED)

    disparity_to_pointcloud(disparity, left_bgr, fx, fy, cx, cy, baseline_m, max_depth=args.max_depth,
                            out_ply="../outputs/pointcloud/cloud.ply")
