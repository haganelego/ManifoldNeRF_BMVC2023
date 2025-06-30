import numpy as np
import os, imageio
from pathlib import Path
import cv2

def sphere_fit(pose_list):
    """
    入力
        point_cloud: 点群のxyz座標のリスト numpyのarray形式で3x3xN行列
    出力
        radius : 近似球の半径 スカラー
        sphere_center : 球の中心座標 xyz numpyのarray
    """

    A_1 = np.zeros((3,3))
    #Aのカッコの中の１項目用に変数A_1をおく
    v_1 = np.array([0.0,0.0,0.0])
    v_2 = 0.0
    v_3 = np.array([0.0,0.0,0.0])
    #ベクトルの１乗、２乗、３乗平均の変数をv_1, v_2, v_3とする
    #１乗、３乗はベクトル、２乗はスカラーとなる

    N = len(pose_list)
    #Nは点群の数

    """シグマの計算"""
    for v in pose_list:
        v_1 += v
        v_2 += np.dot(v, v)
        v_3 += np.dot(v, v) * v

        A_1 += np.dot(np.array([v]).T, np.array([v]))

    v_1 /= N
    v_2 /= N
    v_3 /= N
    A = 2 * (A_1 / N - np.dot(np.array([v_1]).T, np.array([v_1])))
    # 式②
    b = v_3 - v_2 * v_1
    # 式③
    sphere_center = np.dot(np.linalg.inv(A), b)
    # 式①
    radius = (sum(np.linalg.norm(np.array(pose_list) - sphere_center, axis=1))
              /len(pose_list))

    return radius, sphere_center

# This function is taken from InfoNeRF: https://github.com/mjmjeong/InfoNeRF/blob/main/dataset/load_dtu.py
def load_dtu_data(path, train_scene=None, mask_path=None):
    imgdir = os.path.join(path, 'image')
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    imgnames = [f'{int(i.split("/")[-1].split(".")[0]):03d}.png' for i in imgfiles]

    masks = None
    if mask_path is not None:
        cat = imgfiles[0].split('/')[3]
        seen_views  = '_'.join(map(str,train_scene))
        maskdir = os.path.join('./data/DTU_mask/', cat, seen_views)
        maskfiles = [os.path.join(maskdir, i) for i in imgnames] 
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, 0)
    num = imgs.shape[0]
    
    if mask_path is not None:
        masks = [imread(f)[...,0]/255. for f in maskfiles]
        masks = np.stack(masks, 0)

    cam_path = os.path.join(path, "cameras.npz")
    all_cam = np.load(cam_path)

    focal = 0

    coord_trans_world = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )
    coord_trans_cam = np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=np.float32,
            )

    poses = []
    for i in range(num):
        P = all_cam["world_mat_" + str(i)]
        P = P[:3]

        K, R, t = cv2.decomposeProjectionMatrix(P)[:3]
        K = K / K[2, 2]

        focal += (K[0,0] + K[1,1]) / 2

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3, 3] = (t[:3] / t[3])[:, 0]

        scale_mtx = all_cam.get("scale_mat_" + str(i))
        if scale_mtx is not None:
            norm_trans = scale_mtx[:3, 3:]
            norm_scale = np.diagonal(scale_mtx[:3, :3])[..., None]

            pose[:3, 3:] -= norm_trans
            pose[:3, 3:] /= norm_scale

        pose = (
                coord_trans_world
                @ pose
                @ coord_trans_cam
            )
        poses.append(pose[:3,:4])
    
    radius, sphere_center = sphere_fit([pose[:,3] for pose in poses])
    poses = np.stack(poses)
    print('poses shape:', poses.shape)


    focal = focal / num
    H, W = imgs[0].shape[:2]
    print("HWF", H, W, focal)

    return imgs, poses, [H, W, focal], masks, radius, sphere_center