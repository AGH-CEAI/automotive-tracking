"""
Lidar-related helper functions
Adapted from https://github.com/kuangliu/kitti-utils
"""
# import sys
# sys.path.append('.')
# from kitti_util import project_rect_to_image,project_velo_to_ref,project_ref_to_rect

# # get the calibration:
# calibration = utils.Calibration("../datasets/KITTI/training/calib/"+scene_id+".txt")
# import kitti_util as utils

def get_lidar(
        dir='../datasets/KITTI/training/velodyne/0000',
        filename='000000.bin',
        point_cloud_only=False,
        distance_only=False
    ):
    """
    Read a lidar file

    The format is (x,y,z,r)
    """
    import os
    import numpy as np
    
    file = os.path.join(dir, filename)
    # assert os.path.isfile(file)
    if not os.path.isfile(file):
        print(file,' missing!')
        return np.array([])
    if point_cloud_only: # reads only the (x,y,z) coordinates
        return np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:, :3]
    elif distance_only: # reads only the (r) coordinate
        return np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:, 3]
    else: # entire data is read
        return np.fromfile(file, dtype=np.float32).reshape(-1, 4)
    
def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    """ Project lidar points to a monochromatic image """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    img =  np.copy(img)
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc_velo, calib, 0, 0, img_width, img_height, True
    )
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    # draw the points:
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[np.clip(int(640.0 / depth),0,255), :] # we norm the lidar depth to the used colormap
        cv2.circle(
            img,
            (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1]))),
            1,
            color=tuple(color),
            thickness=-1,
        )

    return img

def get_lidar_in_image_fov(
        pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
    ):
    """ Filter lidar points, keeping those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo
