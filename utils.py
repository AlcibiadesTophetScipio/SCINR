import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

def get_norm_grid(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    yy, xx = torch.meshgrid(*coord_seqs, indexing='ij')
    ret = torch.stack([xx, yy], dim=0)
    if flatten:
        ret = ret.view(ret.shape[0],-1)
    return ret

def get_pixel_grid(subsampling_factor):
    """
    Generate target pixel positions according to a subsampling factor, assuming prediction at center pixel.
    """
    pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
    yy, xx = torch.meshgrid(pix_range, pix_range, indexing='ij') # from old source
    # xx, yy = torch.meshgrid(pix_range, pix_range, indexing='ij')
    return subsampling_factor * (torch.stack([xx, yy]) + 0.5)

def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)
    return output

def get_parameter_num(model, check_name=''):
    if len(check_name) !=0 :
        return sum([v.numel() for k,v in model.named_parameters() if check_name in k])

    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total_num, 'trainable': trainable_num}

def xyz_to_rgb(input_tensor):
    _,H,W = input_tensor.shape
    min_xyz = input_tensor.reshape(3,-1).min(dim=-1)[0]
    max_xyz = input_tensor.reshape(3,-1).max(dim=-1)[0]
    t_tensor = input_tensor.permute(1,2,0)
    norm_xyz = (t_tensor - min_xyz)/(max_xyz-min_xyz)

    # From CIE XYZ to RGB Values
    # https://oceanopticsbook.info/view/photometry-and-visibility/from-xyz-to-rgb
    # convert_m = torch.tensor([
    #     [3.2404542, -1.5371385, -0.4985314],
    #     [-0.9692660, 1.8760108, 0.0415560],
    #     [0.0556434, -0.2040259, 1.0572252]
    #  ]).reshape(1,1,3,3).expand(H,W,3,3)
    # rgb_01 = torch.matmul(convert_m, norm_xyz.unsqueeze(-1)).squeeze()
    # rgb_value = (255 * rgb_01).numpy().astype(np.uint8)

    # rgb_01_ = skimage.color.xyz2rgb(norm_xyz)
    # rgb_value = (255 * rgb_01_).astype(np.uint8)

    rgb_value = (255 * norm_xyz.numpy()).astype(np.uint8)
    return rgb_value

def calc_proj(pred_scene_coords_B3HW, gt_pose_inv_B44, intrinsics_B33, depth_min=0.1):
    B, _, H, W = pred_scene_coords_B3HW.shape
    pred_scene_coords_b3n = pred_scene_coords_B3HW.reshape(B,3,-1)
    pred_scene_coords_b4n = to_homogeneous(pred_scene_coords_b3n)

    # Scene coordinates to camera coordinates.
    gt_inv_poses_b34 = gt_pose_inv_B44[:, :3]
    pred_cam_coords_b3n = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b4n)
    
    # Project scene coordinates.
    Ks_b33 = intrinsics_B33
    pred_px_b3n = torch.bmm(Ks_b33, pred_cam_coords_b3n)
    pred_px_b3n[:, 2].clamp_(min=depth_min)
    pred_px_b2n = pred_px_b3n[:, :2] / pred_px_b3n[:, 2, None]

    return pred_px_b2n.reshape(B,2,H,W)

def custor_conv(conv_shape=[5,5]):
    # conv_kernel = torch.nn.Conv2d(3,3,conv_shape,padding=2)
    # w = torch.ones([3,3,*conv_shape])*1/(conv_shape[0]*conv_shape[1])
    conv_kernel = torch.nn.Conv2d(1,1,conv_shape,padding=2)
    w = torch.ones([1,1,*conv_shape])*1/(conv_shape[0]*conv_shape[1])
    conv_kernel.weight = torch.nn.Parameter(w)
    return conv_kernel

def get_nodata_value(scene_name):
    """
    Get nodata value based on dataset scene name.
    """
    if 'urbanscape' in scene_name.lower() or 'naturescape' in scene_name.lower():
        nodata_value = -1
    else:
        raise NotImplementedError
    return nodata_value

def pick_valid_points(coord_input, nodata_value, boolean=False):
    """
    Pick valid 3d points from provided ground-truth labels.
    @param   coord_input   [B, C, N] or [C, N] tensor for 3D labels such as scene coordinates or depth.
    @param   nodata_value  Scalar to indicate NODATA element of ground truth 3D labels.
    @param   boolean       Return boolean variable or explicit index.
    @return  val_points    [B, N] or [N, ] Boolean tensor or valid points index.
    """
    batch_mode = True
    if len(coord_input.shape) == 2:
        # coord_input shape is [C, N], let's make it compatible
        batch_mode = False
        coord_input = coord_input.unsqueeze(0)  # [B, C, N], with B = 1

    val_points = torch.sum(coord_input == nodata_value, dim=1) == 0  # [B, N]
    val_points = val_points.to(coord_input.device)
    if not batch_mode:
        val_points = val_points.squeeze(0)  # [N, ]
    if boolean:
        pass
    else:
        val_points = torch.nonzero(val_points, as_tuple=True)  # a tuple for rows and columns indices
    return val_points


if __name__ == '__main__':
    pixel_grid = get_pixel_grid(8)
    print(pixel_grid.shape)

    input_tensor = torch.randn([8,3,3])
    homo_tenosr = to_homogeneous(input_tensor, dim=1)
    print(homo_tenosr.shape)

    coord_1 = get_norm_grid([128, 128], flatten=False)
    coord_2 = get_norm_grid([128, 128], flatten=True)

    coord_3 = get_norm_grid([40, 128], flatten=False)
    print(coord_3.shape)
    print('Done!')