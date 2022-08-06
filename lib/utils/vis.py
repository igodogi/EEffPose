from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import torch
import numpy as np
import torchvision
import cv2

from core.inference import get_max_preds


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name,
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*heatmap_height,
                           (num_joints+1)*heatmap_width,
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())

    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = heatmap_width * (j+1)
            width_end = heatmap_width * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images(config, input, meta, target, joints_pred, output,
                      prefix):
    if not config.DEBUG.DEBUG:
        return

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints(
            input, meta['joints'], meta['joints_vis'],
            '{}_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_BATCH_IMAGES_PRED:
        save_batch_image_with_joints(
            input, joints_pred, meta['joints_vis'],
            '{}_pred.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps(
            input, target, '{}_hm_gt.jpg'.format(prefix)
        )
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps(
            input, output, '{}_hm_pred.jpg'.format(prefix)
        )


def print_heatmap(heatmap_dir, flag='forward', mode='nearest'):
    # n_width = 16
    n_width = 17

    def get_maxshape(tensor):
        if isinstance(tensor, list):
            shapes = []
            for ts in tensor:
                shapes.append(get_maxshape(ts))
                max_shape = max(shapes)
        elif isinstance(tensor, tuple):
            shapes = []
            for ts in tensor:
                shapes.append(get_maxshape(ts))
                max_shape = max(shapes)
        elif isinstance(tensor, torch.Tensor):
            max_shape = tensor.shape[-2:]

        return max_shape

    def make_tensor(tensor, shape, mode='nearest'):
        if isinstance(tensor, list):
            output = []
            for i, ts in enumerate(tensor):
                output.extend(make_tensor(ts, shape))
        elif isinstance(tensor, tuple):
            output = []
            for i, ts in enumerate(tensor):
                output.extend(make_tensor(ts, shape))
        else:
            out = tensor.clone()
            # out = torch.nn.Sigmoid()(out)
            output = [torch.nn.Upsample(shape, mode=mode)(out)]
        return output

    def save_tensor(tensor, phase='output'):
        if isinstance(tensor, list):
            tensor = torch.cat(tensor, 1)
        for i in range(tensor.shape[0]):
            path, name = heatmap_dir.split('/@/')
            path = os.path.join(path, 'image_' + str(i), phase)
            if not os.path.exists(path):
                os.makedirs(path)
            ts = tensor[i].detach().unsqueeze(1).cpu()
            grid = torchvision.utils.make_grid(ts, n_width, 2, True, pad_value=1)
            grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            grid = cv2.applyColorMap(grid, cv2.COLORMAP_JET)
            file_name = os.path.join(path, name)
            cv2.imwrite(file_name, grid)

    def forward_hook(m, input_inf, output_inf):
        # import torch
        # output = torch.nn.Sigmoid()(output)

        # if isinstance(output_inf, list):
        #     same_layer = torch.nn.Upsample(output_inf[0].shape[2:4], mode='nearest')
        #     output = []
        #     for o, out in enumerate(output_inf):
        #         output.append(same_layer(out))
        #     output = torch.cat(output, 1)
        # elif isinstance(output_inf, tuple):
        #     return
        # else:
        #     output = output_inf.clone()
        try:
            max_shape = get_maxshape(input_inf)
            inputs = make_tensor(input_inf, shape=max_shape, mode=mode)
            save_tensor(inputs, 'input')

            max_shape = get_maxshape(output_inf)
            outputs = make_tensor(output_inf, shape=max_shape, mode=mode)
            save_tensor(outputs, 'output')
        except:
            pass

    def backward_hook(m, input_grad, output_grad):
        # import torch
        # output = torch.nn.Sigmoid()(output)
        # if isinstance(output_grad, list):
        #     same_layer = torch.nn.Upsample(output_grad[0][0].shape[2:4], mode='nearest')
        #     output = []
        #     for o, out in enumerate(output_grad):
        #         output.append(same_layer(out[0]))
        #     output = torch.cat(output, 1)
        # elif isinstance(output_grad, tuple):
        #     return
        # else:
        #     output = output_grad[0].clone()
        #
        # save_tensor(output)
        try:
            max_shape = get_maxshape(input_grad)
            inputs = make_tensor(input_grad, shape=max_shape, mode=mode)
            save_tensor(inputs, 'input')

            max_shape = get_maxshape(output_grad)
            outputs = make_tensor(output_grad, shape=max_shape, mode=mode)
            save_tensor(outputs, 'output')
        except:
            pass

    if flag == 'backward':
        # return
        return backward_hook
    elif flag == 'forward':
        return forward_hook


# def print_heatmap(heatmap_dir, flag='forward'):
#     n_width = 16
#
#     def forward_hook(m, input_inf, output_inf):
#         # import torch
#         # output = torch.nn.Sigmoid()(output)
#
#         if isinstance(output_inf, list):
#             same_layer = torch.nn.Upsample(output_inf[0].shape[2:4], mode='bilinear')
#             output = []
#             for o, out in enumerate(output_inf):
#                 output.append(same_layer(out))
#             output = torch.cat(output, 1)
#         else:
#             output = output_inf.clone()
#
#         for i in range(output.shape[0]):
#             path, name = heatmap_dir.split('/@/')
#             path = os.path.join(path, 'image_' + str(i))
#             if not os.path.exists(path):
#                 os.makedirs(path)
#             try:
#                 out = output[i].detach().unsqueeze(1).cpu()
#                 grid = torchvision.utils.make_grid(out, n_width, 2, True)
#                 grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
#                 file_name = os.path.join(path, name)
#                 cv2.imwrite(file_name, grid)
#             except:
#                 pass
#
#     def backward_hook(m, input_grad, output_grad):
#         # import torch
#         # output = torch.nn.Sigmoid()(output)
#
#         if isinstance(output_grad, list):
#             same_layer = torch.nn.Upsample(output_grad[0][0].shape[2:4], mode='bilinear')
#             output = []
#             for o, out in enumerate(output_grad):
#                 output.append(same_layer(out[0]))
#             output = torch.cat(output, 1)
#         else:
#             output = output_grad[0].clone()
#
#         for i in range(output.shape[0]):
#             path, name = heatmap_dir.split('/@/')
#             path = os.path.join(path, 'image_' + str(i))
#             if not os.path.exists(path):
#                 os.makedirs(path)
#             out = output[i].detach().unsqueeze(1).cpu()
#             grid = torchvision.utils.make_grid(out, n_width, 2, True)
#             grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
#             file_name = os.path.join(path, name)
#             cv2.imwrite(file_name, grid)
#
#     if flag == 'backward':
#         return backward_hook
#     else:
#         return forward_hook


def print_fftmap(heatmap_dir, flag='forward'):
    n_width = 16

    def forward_hook(m, input_inf, output_inf):
        # import torch
        # output = torch.nn.Sigmoid()(output)

        if isinstance(output_inf, list):
            same_layer = torch.nn.Upsample(output_inf[0].shape[2:4], mode='bilinear')
            output = []
            for o, out in enumerate(output_inf):
                # output.append(output_inf[0])
                output.append(same_layer(out))
            output = torch.cat(output, 1)
        else:
            output = output_inf.clone()

        for i in range(output.shape[0]):
            path, name = heatmap_dir.split('/@/')
            path = os.path.join(path, 'image_' + str(i))
            if not os.path.exists(path):
                os.makedirs(path)
            try:
                out = output[i].detach().unsqueeze(1).cpu()
                fft_out = np.fft.fft2(out)
                fshift_out = np.fft.fftshift(fft_out)
                res_out = np.log(np.abs(fshift_out) + 0.000001)
                res_out = torch.tensor(res_out)
                grid = torchvision.utils.make_grid(res_out, n_width, 2, True)
                grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                file_name = os.path.join(path, name)
                cv2.imwrite(file_name, grid)
            except:
                pass

    def backward_hook(m, input_grad, output_grad):
        # import torch
        # output = torch.nn.Sigmoid()(output)

        if isinstance(output_grad, list):
            same_layer = torch.nn.Upsample(output_grad[0][0].shape[2:4], mode='bilinear')
            output = []
            for o, out in enumerate(output_grad):
                output.append(same_layer(out[0]))
            output = torch.cat(output, 1)
        else:
            output = output_grad[0].clone()

        for i in range(output.shape[0]):
            path, name = heatmap_dir.split('/@/')
            path = os.path.join(path, 'image_' + str(i))
            if not os.path.exists(path):
                os.makedirs(path)
            out = output[i].detach().unsqueeze(1).cpu()
            fft_out = np.fft.fft2(out)
            fshift_out = np.fft.fftshift(fft_out)
            res_out = np.log(np.abs(fshift_out) + 0.000001)
            res_out = torch.tensor(res_out)
            grid = torchvision.utils.make_grid(res_out, n_width, 2, True)
            grid = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
            file_name = os.path.join(path, name)
            cv2.imwrite(file_name, grid)

    if flag == 'backward':
        return backward_hook
    else:
        return forward_hook
