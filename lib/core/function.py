from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images

logger = logging.getLogger(__name__)


def train(cfg, train_loader, model, criterion, optimizer,
          epoch, output_dir, tb_log_dir, writer_dict,
          fp16=False, scheduler=None, ema=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    if cfg.WORLD_SIZE > 1:
        torch.distributed.barrier()
    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()
        data_time.update(time.time() - end)

        # compute output
        outputs = model(input)

        if isinstance(outputs, list):
            loss = criterion(outputs[0], target, target_weight)
            for output in outputs[1:]:
                loss += criterion(output, target, target_weight)
            output = outputs[-1]
        else:
            output = outputs
            loss = criterion(output, target, target_weight)

        # compute gradient and do SGD step
        accumulation_steps = cfg.TRAIN.ACCUMULATION_STEPS
        if accumulation_steps != 1:
            # 1 loss regularization
            loss = loss / accumulation_steps
            # 2 back propagation
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            # 3. update parameters of net
            if ((i + 1) % accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()  # update parameters of net
                optimizer.zero_grad()  # reset gradient
                if ema is not None:
                    ema.update()
        else:
            optimizer.zero_grad()
            if fp16:
                optimizer.backward(loss)
            else:
                loss.backward()
            optimizer.step()
            try:
                scheduler.step()
            except:
                pass
            if ema is not None:
                ema.update()

        # measure accuracy and record loss
        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(), target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.PRINT_FREQ == 0 or (i == len(train_loader) - 1):
            msg = f'Train_{cfg.RANK}: [{epoch}][{i}/{len(train_loader)}]\t\t' \
                f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t ' \
                f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                f'Loss {losses.val:.3e} ({losses.avg:.3e})\t' \
                f'Acc {acc.val:.3f} ({acc.avg:.3f})\t'
            logger.info(msg)

        if i == (len(train_loader) - 1):
            prefix = f"{os.path.join(output_dir, 'train')}_{epoch}_{cfg.RANK}"
            stride = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.HEATMAP_SIZE[0]
            save_debug_images(cfg, input, meta, target, pred * stride, output, prefix)

    if writer_dict is not None:
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']
        writer.add_scalar(f'actual_lr/{cfg.RANK}', optimizer.param_groups[0]['lr'], global_steps)
        writer.add_scalar(f'train_loss/{cfg.RANK}', losses.avg, global_steps)
        writer.add_scalar(f'train_acc/{cfg.RANK}', acc.avg, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1
        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()


def valid(cfg, val_loader, val_dataset, model, criterion,
          epoch, output_dir, tb_log_dir, writer_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    all_preds = []
    all_boxes = []
    image_path = []

    idx = 0
    with torch.no_grad():
        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if cfg.TEST.FLIP_TEST:
                input_flipped = input.flip(3)
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(), val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                if cfg.TEST.SHIFT_HEATMAP:
                    output_flipped[:, :, :, 1:] = output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            # compute loss
            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(), target.cpu().numpy())
            acc.update(avg_acc, cnt)

            # make and gather predictions
            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['score'].numpy()
            preds, maxvals = get_final_preds(cfg, output.clone().cpu().numpy(), c, s)
            all_preds.extend(np.concatenate((preds, maxvals), axis=2))
            all_boxes.extend(np.concatenate((c, s, np.prod(s * 200, 1, keepdims=True), score[:, np.newaxis]), axis=1))
            image_path.extend(meta['image'])

            idx += num_images

            # measure elapsed time
            if cfg.WORLD_SIZE > 1:
                torch.distributed.barrier()
            batch_time.update(time.time() - end)
            end = time.time()

            # logger.info
            if i % cfg.PRINT_FREQ == 0 or (i == len(val_loader) - 1):
                # record weights, parameters and gradients of models on each GPUS
                parameter = sum([p.abs().sum() for p in model.parameters()])
                buffer = sum([b.abs().sum() for n, b in model.named_buffers() if 'bn' in n])
                try:
                    weight = sum([w.abs().sum() for w in model.state_dict().values()])
                except AttributeError:
                    weight = sum([w.abs().sum() for w in model.module.state_dict().values()])
                msg = f'Valid_{cfg.RANK}: [{epoch}][{i}/{len(val_loader)}]\t\t' \
                    f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t ' \
                    f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                    f'Loss {losses.val:.3e} ({losses.avg:.3e})\t' \
                    f'Acc {acc.val:.3f} ({acc.avg:.3f})\t' \
                    f'Parameter {parameter: .1f}\t'\
                    f'Buffer {buffer: .1f}\t' \
                    f'Weight {weight: .1f}\t'
                logger.info(msg)

            if i == (len(val_loader) - 1):
                prefix = f"{os.path.join(output_dir, 'val')}_{epoch}_{cfg.RANK}"
                stride = cfg.MODEL.IMAGE_SIZE[0] / cfg.MODEL.HEATMAP_SIZE[0]
                save_debug_images(cfg, input, meta, target, pred * stride, output, prefix)

        all_preds = np.array(all_preds)
        all_boxes = np.array(all_boxes)
        image_path = np.array(image_path)
        perf_indicator = -1
        if not cfg.WORLD_SIZE > 1:
            name_values, perf_indicator = val_dataset.evaluate(cfg, all_preds, output_dir, all_boxes, image_path)
        else:
            np.save(output_dir + '/all_preds_' + str(cfg.RANK), all_preds)
            np.save(output_dir + '/all_boxes_' + str(cfg.RANK), all_boxes)
            np.save(output_dir + '/image_path_' + str(cfg.RANK), image_path)
            if cfg.WORLD_SIZE > 1:
                torch.distributed.barrier()
            all_preds = np.load(output_dir + '/all_preds_0' + '.npy')
            all_boxes = np.load(output_dir + '/all_boxes_0' + '.npy')
            image_path = np.load(output_dir + '/image_path_0' + '.npy')
            for world in range(1, cfg.WORLD_SIZE):
                all_preds = np.append(all_preds, np.load(output_dir + '/all_preds_' + str(world) + '.npy'), 0)
                all_boxes = np.append(all_boxes, np.load(output_dir + '/all_boxes_' + str(world) + '.npy'), 0)
                image_path = np.append(image_path, np.load(output_dir + '/image_path_' + str(world) + '.npy'), 0)
            if cfg.DATASET.DATASET == 'mpii' and cfg.WORLD_SIZE > 1:
                print('As for DistributedSampler, we need sample back the samples and clip the tile.')
                all_preds = all_preds.reshape(cfg.WORLD_SIZE, -1, *all_preds.shape[1:]) \
                    .transpose(1, 0, 2, 3).reshape(-1, *all_preds.shape[1:])
                all_boxes = all_boxes.reshape(cfg.WORLD_SIZE, -1, *all_boxes.shape[1:]) \
                    .transpose(1, 0, 2).reshape(-1, *all_boxes.shape[1:])
                image_path = image_path.reshape(cfg.WORLD_SIZE, -1, *image_path.shape[1:]) \
                    .transpose(1, 0).reshape(-1)
                all_preds = all_preds[:len(val_dataset)]
                all_boxes = all_boxes[:len(val_dataset)]
                image_path = image_path[:len(val_dataset)]
            name_values, perf_indicator = val_dataset.evaluate(cfg, all_preds, output_dir, all_boxes, image_path)

        if cfg.RANK == 0:
            model_name = cfg.MODEL.NAME
            if isinstance(name_values, list):
                for name_value in name_values:
                    _print_name_value(name_value, model_name)
            else:
                _print_name_value(name_values, model_name)

        if writer_dict is not None:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(f'val_loss/{cfg.RANK}', losses.avg, global_steps)
            writer.add_scalar(f'val_acc/{cfg.RANK}', acc.avg, global_steps)
            writer.add_scalar(f'perf_indicator/{cfg.RANK}', perf_indicator, global_steps)
            writer_dict['valid_global_steps'] = global_steps + 1

    return losses.avg, perf_indicator


# markdown format output
def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = 255 * torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.std = 255 * torch.tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, self.next_target_weight, self.next_meta = next(self.loader)
            with torch.cuda.stream(self.stream):
                self.next_input = self.next_input.cuda(non_blocking=True).float().sub_(self.mean).div_(self.std)
                self.next_target = self.next_target.cuda(non_blocking=True)
                self.next_target_weight = self.next_target_weight.cuda(non_blocking=True)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_target_weight = None
            self.next_meta = None
            return

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        target_weight = self.next_target_weight
        meta = self.next_meta
        if input is None or target is None or target_weight is None or meta is None:
            raise StopIteration
        self.preload()
        return input, target, target_weight, meta
