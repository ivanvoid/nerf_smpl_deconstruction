from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn

import cv2

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
# plt.figure()
# plt.imshow(undistort_image)
# plt.show()   


def batch_rodrigues(poses):
    ''' 
        poses: N x 3
    '''
    assert poses.shape == (24,3)

    batch_size = poses.shape[0]
    angle = np.linalg.norm(poses + 1e-8, axis=1, keepdims=True)
    rot_dir = poses / angle

    cos = np.cos(angle)[:, None]
    sin = np.sin(angle)[:, None]

    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    zeros = np.zeros([batch_size, 1])
    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros],
                       axis=1)
    K = K.reshape([batch_size, 3, 3])

    ident = np.eye(3)[None]
    rot_mat = ident + sin * K + (1 - cos) * np.matmul(K, K)

    return rot_mat

def get_rigid_transformation(poses, joints, parents, return_joints=False):
    """
        poses: 24 x 3
        joints: 24 x 3
        parents: 24
    """
    assert poses.shape == (24,3)
    assert joints.shape == (24,3)
    assert parents.shape == (24,)

    rot_mats = batch_rodrigues(poses)

    # obtain the relative joints
    rel_joints = joints.copy()
    rel_joints[1:] -= joints[parents[1:]]

    # create the transformation matrix
    transforms_mat = np.concatenate([rot_mats, rel_joints[..., None]], axis=2)
    padding = np.zeros([24, 1, 4])
    padding[..., 3] = 1
    transforms_mat = np.concatenate([transforms_mat, padding], axis=1)

    # rotate each part
    transform_chain = [transforms_mat[0]]
    for i in range(1, parents.shape[0]):
        curr_res = np.dot(transform_chain[parents[i]], transforms_mat[i])
        transform_chain.append(curr_res)
    transforms = np.stack(transform_chain, axis=0)

    posed_joints = transforms[:, :3, 3].copy()

    # obtain the rigid transformation
    padding = np.zeros([24, 1])
    joints_homogen = np.concatenate([joints, padding], axis=1)
    rel_joints = np.sum(transforms * joints_homogen[:, None], axis=2)
    transforms[..., 3] = transforms[..., 3] - rel_joints
    transforms = transforms.astype(np.float32)

    if return_joints:
        return transforms, posed_joints
    else:
        return transforms
    
def get_bounds(xyz, box_padding=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= box_padding
    max_xyz += box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)
    return bounds

### ###

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d

def project(xyz, K, RT):
    """
    xyz: [N, 3]
    K: [3, 3]
    RT: [3, 4]
    """
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xyz = np.dot(xyz, K.T)
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy

def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box

def sample_ray_h36m(img, msk, K, R, T, bounds, nrays, split, f_sample_ratio=0.3, b_sample_ratio=0.7):
    assert f_sample_ratio + b_sample_ratio == 1

    H, W = img.shape[:2]
    ray_o, ray_d = get_rays(H, W, K, R, T)

    pose = np.concatenate([R, T], axis=1)
    bound_mask = get_bound_2d_mask(bounds, K, pose, H, W)

    if True: #cfg.mask_bkgd:
        img[bound_mask != 1] = 0

    msk = msk * bound_mask
    bound_mask[msk == 100] = 0

    if split == 'train':
        nsampled_rays = 0
        face_sample_ratio = f_sample_ratio
        body_sample_ratio = b_sample_ratio
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []
        coord_list = []
        mask_at_box_list = []

        while nsampled_rays < nrays:
            n_body = int((nrays - nsampled_rays) * body_sample_ratio)
            n_face = int((nrays - nsampled_rays) * face_sample_ratio)
            n_rand = (nrays - nsampled_rays) - n_body - n_face

            # sample rays on body
            coord_body = np.argwhere(msk == 1)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]
            # sample rays on face
            coord_face = np.argwhere(msk == 13)
            if len(coord_face) > 0:
                coord_face = coord_face[np.random.randint(
                    0, len(coord_face), n_face)]
            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]

            if len(coord_face) > 0:
                coord = np.concatenate([coord_body, coord_face, coord], axis=0)
            else:
                coord = np.concatenate([coord_body, coord], axis=0)

            ray_o_ = ray_o[coord[:, 0], coord[:, 1]]
            ray_d_ = ray_d[coord[:, 0], coord[:, 1]]
            rgb_ = img[coord[:, 0], coord[:, 1]]

            near_, far_, mask_at_box = get_near_far(bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            coord_list.append(coord[mask_at_box])
            mask_at_box_list.append(mask_at_box[mask_at_box])
            nsampled_rays += len(near_)

        ray_o = np.concatenate(ray_o_list).astype(np.float32)
        ray_d = np.concatenate(ray_d_list).astype(np.float32)
        rgb = np.concatenate(rgb_list).astype(np.float32)
        near = np.concatenate(near_list).astype(np.float32)
        far = np.concatenate(far_list).astype(np.float32)
        coord = np.concatenate(coord_list)
        mask_at_box = np.concatenate(mask_at_box_list)
    else:
        rgb = img.reshape(-1, 3).astype(np.float32)
        ray_o = ray_o.reshape(-1, 3).astype(np.float32)
        ray_d = ray_d.reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        rgb = rgb[mask_at_box]
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]
        coord = np.argwhere(mask_at_box.reshape(H, W) == 1)

    return rgb, ray_o, ray_d, near, far, coord, mask_at_box

### ###

def crop_mask_edge(msk, border=10):
    msk = msk.copy()
    kernel = np.ones((border, border), np.uint8)
    msk_erode = cv2.erode(msk.copy(), kernel)
    msk_dilate = cv2.dilate(msk.copy(), kernel)
    msk[(msk_dilate - msk_erode) == 1] = 100
    return msk

#####################################

class ZJU_Mocap_Dataset():
    def __init__(self, data_root):
        # Parametes 
        N_RAYS = 32
        self.data_root = data_root

        # load annotations
        annotations_path = os.path.join(data_root, 'annots.npy')
        self.annotations = np.load(
            annotations_path, 
            allow_pickle=True).item()
        
        # prepare cameras
        self.cameras = self.annotations['cams']

        # prepare images and camera idxs
        start_idx = 0
        step = 1 
        end_idx = 100
        view = [0, 6, 12, 18] # idxs of used cameras

        self.num_cams = len(view)

        out = self._process_images(
            data_root, self.annotations['ims'], 
            start_idx, step, end_idx, view)
        self.image_paths, self.mask_paths, self.camera_idxs = out


        # joints and big_A
        self.lbs_root = os.path.join(data_root, 'lbs')
        
        self.joints = np.load(os.path.join(self.lbs_root, 'joints.npy')).astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        
        # TODO: WTF IS THIS?
        self.big_A = self._load_bigpose(self.joints, self.parents)
        
        self.nrays = N_RAYS

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # get image and mask
        image, mask = self._get_image_and_mask(idx)


        # get camera parameters
        camera_idx = self.camera_idxs[idx]
        # INTRINSICS
        # K: Camera matrix
        # D: Distortion coefficients
        camera_matrix = np.array(self.cameras['K'][camera_idx])
        distortion_coefficients = np.array(self.cameras['D'][camera_idx])
        # EXTRINSIC
        # R: rotation matrix
        # T: translation matrix
        R = np.array(self.cameras['R'][camera_idx])
        T = np.array(self.cameras['T'][camera_idx]) / 1000.

        undistort_image = cv2.undistort(image, camera_matrix, distortion_coefficients)
        undistort_mask = cv2.undistort(mask, camera_matrix, distortion_coefficients)
        
        # TODO: scale
        scale = 0.5
        H, W = int(image.shape[0] * scale), int(image.shape[1] * scale)
        img = cv2.resize(image, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # if cfg.mask_bkgd:
        if True:
            img[msk == 0] = 0
        camera_matrix[:2] = camera_matrix[:2] * scale



    
        # read vertices?
        # TODO line 164 in tpose_dataset
        vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        # print(tpose)
        tbounds = get_bounds(tpose)
        tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy')).astype(np.float32)

        img_path = self.image_paths[idx]
        frame_idx = int(os.path.basename(img_path)[:-4])
        wpts, ppts, A, pbw, Rh, Th = self._prepare_input(frame_idx)

        pbounds = get_bounds(ppts)
        wbounds = get_bounds(wpts)
        
        
        rgb, ray_o, ray_d, near, far, coord, mask_at_box = sample_ray_h36m(
            img, msk, camera_matrix, R, T, wbounds, self.nrays, 'train')
        
        orig_msk = msk.copy()
        if True: #cfg.erode_edge:
            orig_msk = crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]
        
        #

        # nerf
        ret = {
            'rgb': rgb,
            'occupancy': occupancy,
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box
        }

        # blend weight
        meta = {
            'A': A,
            'big_A': self.big_A,
            'pbw': pbw,
            'tbw': tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = idx // self.num_cams
        bw_latent_index = idx // self.num_cams

        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': frame_idx,
            'cam_ind': camera_idx
        }
        ret.update(meta)

        return ret

    def _process_images(self, data_root, image_data, start_idx, step, end_idx, view):
        # select number of images from "start_idx" to "end_idx" with step "step"
        image_data_slice = image_data[start_idx:start_idx+step*end_idx]

        image_names = []
        camera_idxs = []
        for im_data in image_data_slice:
            # im_data: one frame from different cameras

            # images
            im_names = np.array(im_data['ims'])
            image_names += [im_names[view]]

            # cameras
            cam_idxs = np.arange(len(im_names))
            camera_idxs += [cam_idxs[view]]

        image_names = np.array(image_names).ravel()
        camera_idxs = np.array(camera_idxs).ravel()

        # combine path
        mask_paths = [os.path.join(data_root, 'mask_cihp', im_name[:-4]+'.png') for im_name in image_names]

        image_paths = [os.path.join(data_root, im_name) for im_name in image_names]

        return image_paths, mask_paths, camera_idxs

    def _load_bigpose(self, joints, parents, angle=30):
        big_poses = np.zeros([len(joints), 3]).astype(np.float32).ravel()
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = get_rigid_transformation(
            big_poses, joints, parents)
        big_A = big_A.astype(np.float32)
        return big_A

    def _get_image_and_mask(self, idx):
        # path
        image_path, mask_path = self.image_paths[idx], self.mask_paths[idx]

        # load
        image = Image.open(image_path)
        mask =  Image.open(mask_path)
        assert image.size == mask.size

        # # scale
        # H,W = image.size
        # new_size = (int(H*scale), int(W*scale))
        
        # image = image.resize(new_size, Image.Resampling.BICUBIC)
        # mask = mask.resize(new_size, Image.Resampling.BICUBIC)

        image = np.array(image)
        mask = (np.array(mask) != 0).astype(np.uint8)

        return image, mask

    def _prepare_input(self, frame_idx, vertices='vertices', params='params'):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(
            self.data_root, vertices,f'{frame_idx}.npy')
        
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, params, f'{frame_idx}.npy')
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = get_rigid_transformation(poses, joints, parents)

        pbw = np.load(os.path.join(self.lbs_root, f'bweights/{frame_idx}.npy'))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, Rh, Th
    

if __name__ == '__main__':
    data_root = '../data/CoreView_387/'

    dataset = ZJU_Mocap_Dataset(data_root)
    print(len(dataset))

    # for i in range(100):
    #     print(dataset[i])
    #     break
    ret = dataset[0]
    keys = ['rgb', 'occupancy', 'ray_o', 'ray_d', 'near', 
            'far', 'mask_at_box', 'A', 'big_A', 'pbw', 'tbw', 
            'pbounds', 'wbounds', 'tbounds', 'R', 'Th', 'H', 
            'W', 'latent_index', 'bw_latent_index', 'frame_index', 'cam_ind']

    for k in keys:
        try:
            print(k, '\t', ret[k].shape)
        except:
            print(ret[k])

# plt.figure();plt.imshow(ret['rgb']);plt.show()   
# plt.figure();plt.imshow(ret['rgb']);plt.show()   