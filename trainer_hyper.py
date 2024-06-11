import logging
import random
import time

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F
from pathlib import Path
import argparse
from distutils.util import strtobool

from utils import get_norm_grid, get_pixel_grid, to_homogeneous, get_parameter_num
from trainer_base import TrainerACE
from config import get_configs

from model.acenet import Regressor as ACENet
from model.base_regressor import BaseRegressor
from model import heads_repo

_logger = logging.getLogger(__name__)

class TrainerHyper(TrainerACE):
    def __init__(self, options):
        super().__init__(options)

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, dataset_id_b1, norm_pos_b2):
        """
        Run one iteration of training, computing the reprojection error and minimising it.
        """
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Reshape to a "fake" BCHW shape, since it's faster to run through the network compared to the original shape.
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        dataset_id_b1HW = dataset_id_b1[None, None, ...].view(-1, 16, 32, 1).permute(0, 3, 1, 2)
        norm_pos_b2HW = norm_pos_b2[None, None, ...].view(-1, 16, 32, 2).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            embedding_repo=None
            if hasattr(self.regressor.heads, 'update_dataEmbedding'):
                embedding_repo = self.regressor.heads.update_dataEmbedding(options.update_num)
            if hasattr(self.regressor.heads, 'generate_params'):
                pred_scene_coords_b3HW = torch.zeros(*[features_bCHW.shape[0], 3, 16, 32]).to(self.device)
                for data_id in torch.unique(dataset_id_b1):
                    net_params = self.regressor.heads.generate_params(
                        dataset_id=data_id,
                        embedding_repo=embedding_repo,
                    )
                    features_bCHW_ = torch.where(dataset_id_b1HW==data_id, features_bCHW, 0)
                    norm_pos_b2HW_ = torch.where(dataset_id_b1HW==data_id, norm_pos_b2HW, 0)
                    pred_scene_coords_b3HW_ = self.regressor.get_scene_coordinates(
                        features_bCHW_,
                        net_params=net_params,
                        norm_pos=norm_pos_b2HW_,
                        dataset_ids=dataset_id_b1,
                    )
                    pred_scene_coords_b3HW = torch.where(
                        dataset_id_b1HW==data_id,
                        pred_scene_coords_b3HW_,
                        pred_scene_coords_b3HW
                    )
            else:
                pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(
                    features_bCHW,
                    dataset_ids=dataset_id_b1HW,
                    norm_pos=norm_pos_b2HW,
                )

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # Make 3D points homogeneous so that we can easily matrix-multiply them.
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

        # Scene coordinates to camera coordinates.
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)

        # Project scene coordinates.
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min. The predicted pixel would be wrong,
        # but that's fine since we mask them out later.
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)

        # Dehomogenise.
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Measure reprojection error.
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        #
        # Compute masks used to ignore invalid pixels.
        #
        # Predicted coordinates behind or close to camera plane.
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        # Very large reprojection errors.
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        # Predicted coordinates beyond max distance.
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max

        # Invalid mask is the union of all these. Valid mask is the opposite.
        invalid_mask_b1 = (invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        loss = 0.0
        if self.options.balance_train:
            loss_list = []
            mask_num_list = []
            avg_list = []
            for data_id in torch.unique(dataset_id_b1):
                dataset_id_mask_b1 = (dataset_id_b1==data_id)
                mask_num_list.append( dataset_id_mask_b1.sum() )
                # Reprojection error for all valid scene coordinates.
                valid_reprojection_error_b1 = reprojection_error_b1[ valid_mask_b1 & dataset_id_mask_b1 ]
                # Compute the loss for valid predictions.
                loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

                # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
                pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
                target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

                # Compute the distance to target camera coordinates.
                loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).sum(dim=1)[invalid_mask_b1 & dataset_id_mask_b1].sum()

                # Final loss is the sum of all 2.
                loss_list.append(loss_valid + loss_invalid)

                avg_list.append((loss_valid+loss_invalid)/dataset_id_mask_b1.sum())

            
            # avg_list = [(i/j) for i,j in zip(loss_list, mask_num_list)]

            loss_avg = sum(loss_list).item() / batch_size
            # co_list = F.normalize(torch.tensor(avg_list), dim=0, p=1)
            # co_list = F.normalize(torch.tensor(avg_list), dim=0, p=2)**2
            # co_list = F.normalize(torch.tensor([i/loss_avg for i in avg_list]),dim=0,p=1)
            # co_list = F.normalize(torch.tensor([i/loss_avg for i in avg_list]),dim=0,p=2)**2

            # co_list = F.softmax(torch.tensor([i/loss_avg for i in avg_list]), dim=0)
            # co_list = F.softmax(torch.tensor(avg_list)/10, dim=0)
            co_list = [(i/loss_avg)**2 for i in avg_list]
            for co, v in zip(co_list, loss_list):
                loss += co*v
            loss /= batch_size
            pass

            # loss_sum = sum(loss_list).item()
            # loss_avg = sum(loss_list).item() / batch_size
            # for i in range(len(loss_list)):
            #     # loss += batch_size/mask_num_list[i] * loss_list[i]
            #     # loss += (1-mask_num_list[i]/batch_size) * loss_list[i]
            #     # loss += (loss_list[i].item()/loss_sum) * loss_list[i]
            #     co = loss_list[i].item()/mask_num_list[i].item()/loss_avg
            #     loss += co**2 * loss_list[i] # avg
            # loss /= batch_size

            
        else:
            # Reprojection error for all valid scene coordinates.
            valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
            # Compute the loss for valid predictions.
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

            # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
            pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
            target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)

            # Compute the distance to target camera coordinates.
            invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
            loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

            # Final loss is the sum of all 2.
            loss = loss_valid + loss_invalid
            loss /= batch_size
            pass

        cls_v = -1.0
        # if hasattr(self.regressor.heads, "dataEmbedding"):
        #     cls_num = self.regressor.heads.dataEmbedding.num_embeddings
        #     cls_label = torch.nn.functional.one_hot(torch.arange(0, cls_num)).to(self.device)
        #     embedding_ids = torch.arange(0, cls_num).to(self.device)
        #     pred_label = self.regressor.heads.pred_cls(embedding_ids)
        #     loss_cls = torch.nn.functional.cross_entropy(pred_label, cls_label.float())
        #     cls_v = loss_cls.item()
        #     loss += loss_cls

        sim_v = -1.0
        if hasattr(self.regressor.heads, "get_cos_similarity"):
            loss_cos=self.regressor.heads.get_cos_similarity(dataset_id_b1)
            sim_v = loss_cos.detach().item()
            loss += 10.0*sim_v
            pass

        # Measure cos similarity
        cos_sim = (1-F.cosine_similarity(pred_cam_coords_b31, target_camera_coords_b31, dim=1)).abs()
        cos_v = cos_sim.masked_select(valid_mask_b1).abs().sum()
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        cos_inv = cos_sim.unsqueeze(1).masked_select(invalid_mask_b11).abs().sum()
        
        # We need to check if the step actually happened, since the scaler might skip optimisation steps.
        old_optimizer_step = self.optimizer._step_count

        # Optimization steps.
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()

        # Debug
        # max_grad_value, max_grad_norm= 0.0, 0.0
        # for param in self.regressor.heads.parameters():
        #     if torch.isinf(param).any() or torch.isinf(param).any():
        #         exit(1)
        #     if param.grad is None:
        #         continue
        #     if param.grad.abs().max()>max_grad_value:
        #         max_grad_value=param.grad.abs().max()
        #     if param.grad.norm(p=2)>max_grad_norm:
        #         max_grad_norm=param.grad.norm(p=2)
        # if torch.isnan(loss):
        #     print(self.iteration, max_grad_norm, max_grad_value)
        #     exit(1)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            # Print status.
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # median_depth = float(pred_cam_coords_b31[:, 2].median())

            cos_v = (cos_v.item()+1e-8) / (valid_mask_b1.sum()+1e-8)
            cos_inv = (cos_inv.item()+1e-8) / (invalid_mask_b11.sum()+1e-8)
            _logger.info(f'Iteration: {self.iteration:6d} / Epoch {self.epoch:03d}|{self.options.epochs:03d}, lr: {self.optimizer.param_groups[0]["lr"]:.6f}, '
                         f'Loss: {loss:.2f}, Valid: {fraction_valid * 100:.2f}%, Cos_v: {cos_v:.2f}, Cos_inv: {cos_inv:.2f}, '
                         f'Cls:{cls_v:.2f}, Sim: {sim_v:.4f}, '
                        #  f'Grad_v: {max_grad_value:.2f}, Grad_norm: {max_grad_norm:.2f} '
                         f'Time: {time_since_start:.2f}s')


        if not hasattr(self.scheduler, 'total_steps'):
            return
        # Only step if the optimizer stepped and if we're not over-stepping the total_steps supported by the scheduler.
        if old_optimizer_step < self.optimizer._step_count < self.scheduler.total_steps:
            self.scheduler.step()

if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    options = get_configs()
    print(options)

    if options.regressor_select == 'ace':
        options.Regressor = ACENet
    elif options.regressor_select == 'base':
        options.Regressor = BaseRegressor
    else:
        raise Exception("Regressor error!")
    
    options.Head = heads_repo[options.head_mod]

    trainer = TrainerHyper(options)
    trainer.train()