from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import valid
from utils.utils import parse_args
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def main():
    # Update config
    global args
    args = parse_args()
    args.distributed = False
    args.world_size = 1
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
    update_config(cfg, args)

    # Creat logger
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'test')
    if cfg.RANK == 0:
        logger.info(f'args: {pprint.pformat(args)}')
        logger.info(f'cfg: {cfg}')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    cudnn.enabled = cfg.CUDNN.ENABLED
    torch.cuda.set_device(cfg.GPUS[cfg.RANK])

    # Reproducibility
    import random
    import numpy as np
    if args.deterministic:
        cudnn.enabled = False
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(cfg.RANK)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.RANK)
            torch.cuda.manual_seed_all(cfg.RANK)
            logger.info('cuda deterministic')
        os.environ['PYTHONHASHSEED'] = str(cfg.RANK)
        random.seed(cfg.RANK)
        np.random.seed(cfg.RANK)
        logger.info('Make model deterministic!')

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # Get model
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=False)

    # FP16
    if cfg.FP16.ENABLED:
        from fp16_utils.fp16util import network_to_half
        model = network_to_half(model)

    # SYNC_BN
    if cfg.MODEL.SYNC_BN and not args.distributed:
        logger.info('Warning: Sync BatchNorm is only supported in distributed training.')

    # DataParallel or DistributedDataParallel
    if args.distributed:
        if cfg.MODEL.SYNC_BN:
            # import apex
            # model = apex.parallel.convert_syncbn_model(model)
            # logger.info('Using apex.parallel.convert_syncbn_model to Sync BatchNorm.')
            # from sync_batchnorm import convert_model
            # model = convert_model(model)
            # logger.info('Using sync_batchnorm by @vacancy.')
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            logger.info('Using torch.nn.SyncBatchNorm.convert_sync_batchnorm to Sync BatchNorm.')
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if len(cfg.GPUS) > 1:
            torch.cuda.set_device(cfg.GPUS[cfg.RANK])
            model.cuda(cfg.GPUS[cfg.RANK])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            # args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.GPUS[cfg.RANK]],
                output_device=cfg.GPUS[cfg.RANK],
                find_unused_parameters=False
            )
    elif len(cfg.GPUS) > 1:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    if cfg.RANK == 0:
        # copy model file
        this_dir = os.path.dirname(__file__)
        shutil.copytree(os.path.join(this_dir, '../experiments'),
                        os.path.join(final_output_dir, 'code', 'experiments'))
        shutil.copytree(os.path.join(this_dir, '../lib'),
                        os.path.join(final_output_dir, 'code', 'lib'))
        shutil.copytree(os.path.join(this_dir, '../tools'),
                        os.path.join(final_output_dir, 'code', 'tools'))
        shutil.copytree(os.path.join(this_dir, '../visualization'),
                        os.path.join(final_output_dir, 'code', 'visualization'))

        # write the model graph and info down
        logger.info(pprint.pformat(model))
        # dump_input = torch.zeros((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])).cuda()
        # # writer_dict['writer'].add_graph(model, (dump_input, ))
        # logger.info(get_model_summary(model, dump_input))
        # # logger.info('The Parameters and MACs above are wrong, be in accordance with the following by thop!')
        # # # model summary by thop
        # # from thop import profile, profile_origin, clever_format
        # # macs2, params2 = profile_origin(model, inputs=(dump_input, ), verbose=False)
        # # macs2, params2 = clever_format([macs2, params2], "%.3f")
        # # logger.info('#' * 96)
        # # logger.info('model summary by thop')
        # # logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs2))
        # # logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params2))
        # del dump_input

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # load checkpoint if needed/ wanted
    if cfg.TEST.MODEL_FILE:
        checkpoint = torch.load(cfg.TEST.MODEL_FILE, map_location='cpu')
        if 'state_dict' in checkpoint.keys():
            checkpoint = checkpoint['state_dict']
        logger.info('=> Testing model from {}'.format(cfg.TEST.MODEL_FILE))
        try:
            ret = model.load_state_dict(checkpoint, strict=True)
            logger.info(ret)
        except:
            ret = model.module.load_state_dict(checkpoint, strict=True)
            logger.info(ret)
        logger.info("=> Testing checkpoint '{}'".format(cfg.TEST.MODEL_FILE))

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, False,
    #                                                  transforms.Compose([transforms.ToTensor(), normalize, ]))
    dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
                                                     transforms.Compose([transforms.ToTensor(), normalize, ]))

    sampler = None
    num = len(cfg.GPUS)
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        num = 1

    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * num,
                                         shuffle=False, num_workers=cfg.WORKERS, pin_memory=True,
                                         sampler=sampler, worker_init_fn=seed_worker)

    # evaluate
    with torch.no_grad():
        # torch.no_grad() impacts the autograd engine and deactivate it.
        # It will reduce memory usage and speed up computations,
        # but you won’t be able to backprop (which you don’t want in an eval script).
        import time
        begin_time = time.time()
        val_loss, perf_indicator = valid(cfg, loader, dataset, model, criterion,
                                         -1, final_output_dir, tb_log_dir)
        end_time = time.time()
        if cfg.RANK == 0:
            print('Testing: {}s spent.'.format(str(end_time - begin_time)))
        return val_loss, perf_indicator


if __name__ == '__main__':
    main()
