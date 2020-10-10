import cv2
import numpy as np
import open3d as o3d

# MEMO: axis mappings
# width, height, depth == x, y, z
# img.shape == (y, x)
# pcd.shape == (x, y, z)


def img2pcd(img, target_depth=2.0, target_width=None):
    if target_width is None:
        target_width = img.shape[1]
    target_height = int(img.shape[0] / img.shape[1] * target_width)

    img_gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)

    z = (1 - img_gray / 255) * target_depth
    x, y = np.meshgrid(
        np.linspace(0, target_width, z.shape[1]),
        np.linspace(0, target_height, z.shape[0])
    )
    pcd = np.concatenate([
        x[:, :, None],
        y[:, :, None],
        z[:, :, None],
    ], axis=2)
    return pcd


if __name__ == "__main__":
    img = cv2.imread('data/500yen.jpg')
    pcd = img2pcd(img, target_depth=100)
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.reshape(-1, 3))

    o3d.visualization.draw_geometries([o3d_pcd])
