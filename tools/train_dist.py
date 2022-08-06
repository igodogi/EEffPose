from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
import dataset
import models

from config import cfg
from config import update_config
from core.loss import JointsMSELoss, JointsOHKMMSELoss
from core.function import train
from core.function import valid
from utils.utils import parse_args
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary


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
    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir, filename_suffix=str(cfg.RANK)),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
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
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(cfg, is_train=True)

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
        # shutil.copytree(os.path.join(this_dir, '../demo'),
        #                 os.path.join(final_output_dir, 'code', 'demo'))
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
    if cfg.LOSS.USE_OHKM:
        criterion = JointsOHKMMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    else:
        criterion = JointsMSELoss(
            use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
        ).cuda()
    optimizer = get_optimizer(cfg, model)
    if cfg.FP16.ENABLED:
        from fp16_utils.fp16_optimizer import FP16_Optimizer
        optimizer = FP16_Optimizer(
            optimizer, verbose=False,
            static_loss_scale=cfg.FP16.STATIC_LOSS_SCALE,
            dynamic_loss_scale=cfg.FP16.DYNAMIC_LOSS_SCALE
        )
        opt = optimizer.optimizer
    else:
        opt = optimizer

    # lr_scheduler
    # lr_scheduler = CyclicLRWithRestarts(opt, cfg.TRAIN.BATCH_SIZE_PER_GPU, cfg.TRAIN.END_EPOCH,
    #                                     restart_period=5, t_mult=1.2, policy="cosine")
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.TRAIN.LR, epochs=40, steps_per_epoch=1)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.TRAIN.LR, epochs=40, steps_per_epoch=len(train_loader))
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', cfg.TRAIN.LR_FACTOR, patience=5)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', cfg.TRAIN.LR_FACTOR, patience=10)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', cfg.TRAIN.LR_FACTOR,
    #                                                           patience=10, cooldown=5)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #     opt, cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR, last_epoch=begin_epoch
    # )

    # load checkpoint if needed/ wanted
    best_perf = -1
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    # checkpoint_file = os.path.join(final_output_dir, 'checkpoint_.pth')
    checkpoint_file = 'output/coco/pose_effpnet/effpnetp2_384x288_adam_lr1e-3/2021-09-03-23-29/checkpoint_-5.pth'
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info(f"=> loading checkpoint '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        best_perf = checkpoint['perf']
        begin_epoch = checkpoint['epoch']
        writer_dict['train_global_steps'] = begin_epoch
        writer_dict['valid_global_steps'] = begin_epoch
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        logger.info(f"=> Auto_resume from checkpoint '{checkpoint_file}' (epoch {checkpoint['epoch']})")

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # original
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
                                                           transforms.Compose([transforms.ToTensor(), normalize, ]))
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
                                                           transforms.Compose([transforms.ToTensor(), normalize, ]))

    train_sampler = valid_sampler = None
    num = len(cfg.GPUS)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset, shuffle=False)
        num = 1

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * num,
                                               shuffle=not(train_sampler), num_workers=cfg.WORKERS, pin_memory=True,
                                               sampler=train_sampler, worker_init_fn=seed_worker)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * num,
                                               shuffle=False, num_workers=cfg.WORKERS, pin_memory=True,
                                               sampler=valid_sampler, worker_init_fn=seed_worker)

    # test lr
    if args.test_lr:
        from lib.utils.lr_test import running
        logger.info("==>Testing lr")
        running(cfg, train_loader, model, criterion, optimizer)
        return

    # train
    multiplier = 1.0
    warmup_epoches = cfg.TRAIN.WARMUP_EPOCHES
    base_lrs = [cfg.TRAIN.LR]
    if begin_epoch == 0 and cfg.TRAIN.WARMUP_EPOCHES > 0:
        begin_epoch = -cfg.TRAIN.WARMUP_EPOCHES
    end_epoch = cfg.TRAIN.END_EPOCH
    for epoch in range(begin_epoch, end_epoch):
        if epoch < 0:
            # epoch_now = float(warmup_epoches + epoch)
            epoch_now = float(warmup_epoches + epoch + 1)
            if multiplier == 1.0:
                warmup_lrs = [base_lr * (epoch_now / warmup_epoches)
                             for base_lr in base_lrs]
            else:
                warmup_lrs = [base_lr * ((multiplier - 1.) * epoch_now / warmup_epoches + 1.)
                             for base_lr in base_lrs]
            for param_group, warmup_lr in zip(opt.param_groups, warmup_lrs):
                param_group['lr'] = warmup_lr
        elif epoch == 0:
            for param_group, base_lr in zip(opt.param_groups, base_lrs):
                param_group['lr'] = base_lr

        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()
        begin_time = time.time()
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer,
              epoch, final_output_dir, tb_log_dir, writer_dict,
              scheduler=None, fp16=cfg.FP16.ENABLED)
              # scheduler=lr_scheduler, fp16=cfg.FP16.ENABLED)
        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()
        end_time = time.time()
        logger.info(f'Training @ epoch {epoch}: {end_time - begin_time}s spent.')

        # evaluate on validation set
        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()
        begin_time = time.time()
        val_loss, perf_indicator = valid(
            cfg, valid_loader, valid_dataset, model, criterion,
            epoch, final_output_dir, tb_log_dir, writer_dict
        )
        if cfg.WORLD_SIZE > 1:
            torch.distributed.barrier()
        end_time = time.time()
        logger.info(f'Testing @ epoch {epoch}: {end_time - begin_time}s spent.')

        # remember best model and save checkpoint, only for single-GPU format
        if cfg.RANK == 0:
            # perf_indicator = val_loss
            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False
            try:
                # try to make module.xx.weight/bias to xx.weight/bias for single-GPU load, when DDP/DataParallel
                model_state_dict = model.module.state_dict()
            except:
                # save xx.weight/bias directly, excepting single-GPU/CPU trainning
                model_state_dict = model.state_dict()

            if best_model or (epoch % 5 == 0):
                save_checkpoint(
                    {'epoch': epoch + 1,
                     'model': cfg.MODEL.NAME,
                     'state_dict': model_state_dict,
                     'perf': perf_indicator,
                     'optimizer': optimizer.state_dict(),
                     'lr_scheduler': lr_scheduler.state_dict(),
                     },
                    best_model,
                    final_output_dir,
                    f'checkpoint_{str(epoch)}.pth'
                )
                logger.info(f'=> saving checkpoint to {final_output_dir}')
                if best_model:
                    logger.info('=> saving best_model.')
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(-perf_indicator)
        else:
            lr_scheduler.step(epoch)
        logger.info(f"LR now is: {optimizer.param_groups[0]['lr']}")

    if cfg.RANK == 0:
        final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
        logger.info(f'=> saving final model state to {final_model_state_file}')
        torch.save(model_state_dict, final_model_state_file)


if __name__ == '__main__':
    main()
