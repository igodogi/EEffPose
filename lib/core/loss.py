from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8, ohm='ohpm'):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk
        self.ohm = ohm

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def ohem(self, loss):
        ohem_loss = 0.
        for i in range(loss.size()[1]):
            sub_loss = loss[:, i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohem_loss += torch.sum(tmp_loss) / self.topk
        ohem_loss /= loss.size()[1]
        return ohem_loss, topk_idx

    def ohmm(self, loss):
        loss = loss.reshape(loss.shape[0] * loss.shape[1], -1).mean(1)
        loss_mask = loss > loss.median()
        ohmm_loss = (loss*loss_mask).sum() / loss_mask.sum()
        return ohmm_loss

    def ohpm(self, loss):
        loss = loss.flatten()
        loss_mask = loss > loss.median()
        ohpm_loss = (loss*loss_mask).sum() / loss_mask.sum()
        return ohpm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        if self.ohm == 'ohkm':
            loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
            loss = torch.cat(loss, dim=1)
            return self.ohkm(loss)
        elif self.ohm == 'ohem':
            loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
            loss = torch.cat(loss, dim=1)
            return self.ohem(loss)
        elif self.ohm == 'ohmm':
            loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
            loss = torch.cat(loss, dim=1)
            return self.ohmm(loss)
        elif self.ohm == 'ohpm':
            loss = torch.cat(loss, dim=1)
            return self.ohpm(loss)


if __name__ == '__main__':
    batch_size = 16
    output = torch.rand((batch_size, 17, 64, 64))
    target = torch.rand((batch_size, 17, 64, 64))
    target_weight = torch.ones((batch_size, 17, 1))
    mse_loss = JointsMSELoss(True)
    mse_loss(output, target, target_weight)
    ohkm_loss = JointsOHKMMSELoss(True)
    ohkm_loss(output, target, target_weight)
