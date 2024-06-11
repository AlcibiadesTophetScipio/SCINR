import logging
import random
import time
from pathlib import Path
import argparse
from distutils.util import strtobool

# from base_trainer import TrainerACE
# from cls_trainer import TrainerCLS
# from linear_trainer import TrainerLinear
# from model.acenet import Regressor as ACENet
# from model.base_regressor import BaseRegressor
# from model import heads_repo

def _strtobool(x):
    return bool(strtobool(x))

def get_configs():
    parser = argparse.ArgumentParser(
        description='Fast training of a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--scenes', nargs='+', type=Path, default=[
        "datasets/Cambridge/GreatCourt",
        "datasets/Cambridge/KingsCollege",
        "datasets/Cambridge/OldHospital",
        "datasets/Cambridge/ShopFacade",
        "datasets/Cambridge/StMarysChurch",
    ],
    help='path to a scene in the dataset folder, e.g. "datasets/Cambridge/GreatCourt"')

    parser.add_argument('--output_map_file', type=Path, default="output/temp/cambridge.pt",
                        help='target file for the trained network')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')

    parser.add_argument('--learning_rate_min', type=float, default=0.0005,
                        help='lowest learning rate of 1 cycle scheduler')

    parser.add_argument('--learning_rate_max', type=float, default=0.005,
                        help='highest learning rate of 1 cycle scheduler')

    parser.add_argument('--training_buffer_size', type=int, default=800000,
                        help='number of patches in the training buffer')

    parser.add_argument('--samples_per_image', type=int, default=1024,
                        help='number of patches drawn from each image when creating the buffer')

    parser.add_argument('--batch_size', type=int, default=5120,
                        help='number of patches for each parameter update (has to be a multiple of 512)')

    parser.add_argument('--epochs', type=int, default=16,
                        help='number of runs through the training buffer')

    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')

    parser.add_argument('--repro_loss_soft_clamp_min', type=int, default=1,
                        help='minimum value of the soft clamping threshold when using a schedule')

    parser.add_argument('--use_half', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='train with half precision')

    parser.add_argument('--use_aug', type=_strtobool, default=True,
                        help='Use any augmentation.')

    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')

    parser.add_argument('--aug_scale', type=float, default=1.5,
                        help='max scale factor')

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')

    parser.add_argument('--repro_loss_type', type=str, default="dyntanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')

    parser.add_argument('--repro_loss_schedule', type=str, default="circle", choices=['circle', 'linear'],
                        help='How to decrease the softclamp threshold during training, circle is slower first')

    parser.add_argument('--depth_min', type=float, default=0.1,
                        help='enforce minimum depth of network predictions')

    parser.add_argument('--depth_target', type=float, default=10,
                        help='default depth to regularize training')

    parser.add_argument('--depth_max', type=float, default=1000,
                        help='enforce maximum depth of network predictions')

    # Clustering params, for the ensemble training used in the Cambridge experiments. Disabled by default.
    parser.add_argument('--num_clusters', type=int, default=None,
                        help='split the training sequence in this number of clusters. disabled by default')

    parser.add_argument('--cluster_idx', type=int, default=None,
                        help='train on images part of this cluster. required only if --num_clusters is set.')

    # Params for the visualization. If enabled, it will slow down training considerably. But you get a nice video :)
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')

    parser.add_argument('--render_map_error_threshold', type=int, default=10,
                        help='reprojection error threshold for the visualisation in px')

    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')

    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')
    
    parser.add_argument('--head_mod', type=str, default="RushH",
                        # choices=["RushH"],
                        help='Head model selection.')
    parser.add_argument('--regressor_select', type=str, default="ace",
                        # choices=['ace', 'base'],
                        help='Head model selection.')
    parser.add_argument('--milestones', nargs='+', type=int, 
                        default=[60,],
                        help='multi-step lr scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='multi-step lr scheduler gamma.')
    parser.add_argument('--balance_train', type=_strtobool, default=False,
                        help='Training with a re-balanced method.')
    parser.add_argument('--update_num', type=int, default=5,
                        help='to specify the num of embedding in update.')
    options = parser.parse_args()

    return options

if __name__ == '__main__':
    options = get_configs()
    print(options)