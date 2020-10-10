import cv2
import numpy as np

# MEMO: axis mappings
# width, height, depth == x, y, z
# img.shape == (y, x)
# pcd.shape == (x, y, z)


def img2pcd(img, target_depth=1.0, target_width=None):
    if target_width is None:
        target_width = img.shape[1]
    target_height = int(img.shape[0] / img.shape[1] * target_width)

    img_gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2GRAY)

    z = (1 - cv2.flip(img_gray, 1) / 255) * target_depth
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


def plane2sphere(pcd_plane, radius=50, bottom_hole_radius=30, thickness=2):

    # apply bottom hole
    r_max = int(
        pcd_plane.shape[0] * (1 - np.arcsin(bottom_hole_radius / radius) / np.pi))
    target_pcd_plane = pcd_plane[:r_max].copy()

    y_max = np.max(pcd_plane[:, 0, 1])
    x_max = np.max(pcd_plane[0, :, 0])

    def mapper(xyz):
        x, y, z = xyz
        p = y/y_max*np.pi
        t = x/x_max*2*np.pi
        external_radius = radius + \
            z * thickness if y < y_max - 1 else radius + thickness
        internal_radius = radius
        ret_value = np.array([
            # external xyz
            external_radius*np.cos(t)*np.sin(p),
            external_radius*np.cos(p),
            external_radius*np.sin(t)*np.sin(p),
            # internal xyz
            internal_radius*np.cos(t)*np.sin(p),
            internal_radius * np.cos(p) if y < y_max - \
            1 else external_radius*np.cos(p),
            internal_radius*np.sin(t)*np.sin(p),
        ])
        return ret_value

    umapper = np.vectorize(mapper, signature='(3)->(6)')
    return umapper(target_pcd_plane.reshape(-1, 3))


if __name__ == "__main__":
    import open3d as o3d

    img = cv2.imread('data/500yen.jpg')
    plane_pcd = img2pcd(img)

    # visualize
    # o3d_pcd = o3d.geometry.PointCloud()
    # o3d_pcd.points = o3d.utility.Vector3dVector(plane_pcd.reshape(-1, 3))
    # o3d.visualization.draw_geometries([o3d_pcd])

    sphere_pcd = plane2sphere(plane_pcd, thickness=5)
    print(sphere_pcd.shape)

    # visualize
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(
        sphere_pcd[:, :3].reshape(-1, 3))
    o3d.visualization.draw_geometries([o3d_pcd])
