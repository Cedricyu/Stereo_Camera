import cv2
import numpy as np
import os
import open3d as o3d

def save_pointcloud_ply(path, points, colors, binary=True, bgr=True):
    """
    points: (N,3) float32, in meters (或標定單位)
    colors: (N,3) uint8, BGR 或 RGB
    binary: True=Binary Little Endian, False=ASCII
    bgr:    True 時自動轉成 RGB
    """
    pts = np.asarray(points, dtype=np.float32)
    cols = np.asarray(colors, dtype=np.uint8)
    assert pts.shape[0] == cols.shape[0] and pts.shape[1] == 3 and cols.shape[1] == 3

    # 過濾非有限值
    finite = np.isfinite(pts).all(axis=1)
    pts, cols = pts[finite], cols[finite]

    # BGR -> RGB（PLY 習慣用 RGB）
    if bgr:
        cols = cols[:, ::-1]

    n = pts.shape[0]
    header = (
        "ply\n"
        f"format {'binary_little_endian' if binary else 'ascii'} 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    if binary:
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            rec = np.empty(n, dtype=[("x","<f4"),("y","<f4"),("z","<f4"),
                                     ("red","u1"),("green","u1"),("blue","u1")])
            rec["x"], rec["y"], rec["z"] = pts[:,0], pts[:,1], pts[:,2]
            rec["red"], rec["green"], rec["blue"] = cols[:,0], cols[:,1], cols[:,2]
            rec.tofile(f)
    else:
        with open(path, "w") as f:
            f.write(header)
            for (x,y,z),(r,g,b) in zip(pts, cols):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def visualize_pointcloud(points, colors, show_axis=True):
    """
    points: (N,3) float32
    colors: (N,3) uint8 or float32 (0~255 or 0~1)
    show_axis: 顯示原點座標軸
    """
    # 若顏色是 uint8，轉成 float
    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0

    # 建立 open3d 點雲物件
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)   # (N,3)
    pcd.colors = o3d.utility.Vector3dVector(colors)   # (N,3), float in [0,1]

    geometries = [pcd]

    if show_axis:
        # 建立座標軸：長度為 0.1 公尺
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        geometries.append(axis)

    # 顯示
    o3d.visualization.draw_geometries(
        geometries,
        window_name='Stereo PointCloud',
        width=640, height=480,
        zoom=1.0,
        front=[0.0, 0.0, -1.0],  # 從 +z 看向原點（相機視角）
        lookat=[0.0, 0.0, 0.0],
        up=[0.0, -1.0, 0.0]      # y-down → y-up 換成 [0,1,0]
    )

def compute_pointcloud(disparity, Q, color_bgr, z_min=0.3, z_max=10.0):
    """
    disparity: float32, in pixels (已除以16)
    Q: from cv2.stereoRectify
    color_bgr: 校正後對齊的左影像 (H,W,3)
    z_min: 最小可接受深度（排除視差太小導致的爆炸值）
    z_max: 最大可接受深度（排除過遠雜訊）
    """
    points_3D = cv2.reprojectImageTo3D(disparity, Q)  # (H,W,3)
    Z = points_3D[:, :, 2]
    print("Q: ",Q)
    print(f"[Q] fx = {Q[2,3]:.2f}, baseline = {1/Q[3,2]:.4f} m")
    print(f"[PointCloud] Z range (raw): {np.nanmin(Z):.3f} ~ {np.nanmax(Z):.3f} m")

    # 遮罩條件：視差 > 0、Z 合理、不是 NaN/Inf
    mask = (disparity > 0) & np.isfinite(Z) & (Z > z_min)
    if z_max is not None:
        mask &= Z < z_max

    # 濾出有效點
    pts = points_3D[mask].astype(np.float32)
    cols = color_bgr[mask].astype(np.uint8)

    if pts.shape[0] > 0:
        z_valid = pts[:, 2]
        print(f"[PointCloud] 有效點數: {pts.shape[0]}")
        print(f"[PointCloud] Z min={z_valid.min():.3f}, max={z_valid.max():.3f}, median={np.median(z_valid):.3f} m")
    else:
        print("[PointCloud] ⚠️ 無有效點")

    return pts, cols

