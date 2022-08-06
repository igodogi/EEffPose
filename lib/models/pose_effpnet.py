# -*-coding:utf-8-*-
import os
import copy
import math
import logging
import torch
import numpy as np
from torch import nn
# from tools import _init_paths

from models.effdet.config.model_config import get_efficientdet_config
from models.effdet.efficientdet import EfficientDet

from models.effpnet.model import HighResolutionBlock, TotalFusion
from models.effpnet.model import FinalAttentionHead, FinalSumHead
from models.effpnet.model import _init_weight, _init_weight_alt

logger = logging.getLogger(__name__)


class EffPNet(EfficientDet):
    def __init__(self, config, pretrained_backbone=True, alternate_init=False, **kwargs):
        super(EffPNet, self).__init__(config, pretrained_backbone, alternate_init)
        cfg = kwargs['cfg']
        self.pretrained_layers = cfg.MODEL.EXTRA.PRETRAINED_LAYERS
        self.backbone_type = cfg.MODEL.BACKBONE_TYPE
        self.backbone_recurrent = cfg.MODEL.BACKBONE_REC
        self.fpn_type = cfg.MODEL.FPN_TYPE
        self.fuse_dense = cfg.MODEL.FUSE_DENSE
        self.final_type = cfg.MODEL.FIINAL_TYPE
        self.freeze_backbone = cfg.MODEL.FREEZE_BACKBONE
        self.freeze_stages = cfg.MODEL.FREEZE_STAGES
        self.num_joints = cfg.MODEL.NUM_JOINTS

        self.backbone.conv_stem.stride = (1, 1)
        if self.backbone_type == 'Ours':
            hr_branches = nn.ModuleList()
            hr_backbone = nn.ModuleList([
                self.backbone.conv_stem,
                self.backbone.bn1,
                self.backbone.act1,
            ])
            for idx, block in enumerate(self.backbone.blocks):
                hr_backbone.append(HighResolutionBlock(hr_branches, block, recurrent=self.backbone_recurrent))
                # if True:
                if idx + 1 in list(self.backbone._stage_out_idx.keys())[:-1]:
                    hr_branches.append(block[-1])
                    hr_backbone[-1].make_fuse_layer()
                if self.fuse_dense:
                    # if self.fuse_dense and (idx + 1) != len(self.backbone.blocks):
                    hr_backbone[-1].make_fuse_layer()

            self.hr_backbone = nn.Sequential(*hr_backbone)
            del block, hr_branches, hr_backbone
        else:
            self.hr_backbone = self.backbone

        in_channels = [feature['num_chs'] for feature in self.feature_info][:3]
        # out_channels = [64] * len(in_channels)
        out_channels = [config['fpn_channels']] * len(in_channels)
        if self.fpn_type == 'Ours':
            # drop_path_rates = [0.1] * len(in_channels)
            drop_path_rates = [config.backbone_args.drop_path_rate] * len(in_channels)
            # drop_path_rates = [IR.drop_path_rate for IR in self.backbone.blocks[-1]] * len(in_channels)
            self.fpn = nn.Sequential(
                TotalFusion(in_channels, out_channels, drop_path_rates),
                *[
                    TotalFusion(out_channels, out_channels, drop_path_rates)
                    # for _ in range(2)
                    for _ in range(config['fpn_cell_repeats'] - 1)
                ],
            )
        else:
            in_channels = [feature['num_chs'] for feature in self.feature_info]
            out_channels = [config['fpn_channels']] * len(in_channels)

        if self.final_type == 'Ours':
            self.final_layer = FinalAttentionHead(
                out_channels,
                self.num_joints,
            )
        else:
            self.final_layer = FinalSumHead(
                out_channels,
                self.num_joints,
            )

        for n, m in self.named_modules():
            if 'backbone' not in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

        for n, m in self.named_modules():
            if 'fuse_layers' in n:
                if alternate_init:
                    _init_weight_alt(m, n)
                else:
                    _init_weight(m, n)

        # for n, m in self.named_modules():
        #     from timm.models.efficientnet_blocks import SqueezeExcite
        #     if isinstance(m, SqueezeExcite) and 'backbone' not in n:
        #         if alternate_init:
        #             _init_weight_alt(m, n)
        #         else:
        #             _init_weight(m, n)

    def forward(self, inputs):
        features = self.hr_backbone(inputs)

        features = self.fpn(features)

        return [self.final_layer(features)]

    def load_ckpt(self, pretrained=''):
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, map_location='cpu')
            logger.info('=> loading pretrained model {} for {}'.format(pretrained, self.pretrained_layers))
            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers or (self.pretrained_layers[0] is '*'):
                    need_init_state_dict[name] = m

            ret = self.load_state_dict(need_init_state_dict, strict=False)
            logger.info(ret)

    def visualization(self, ):
        from utils.vis import print_heatmap, print_fftmap
        global_n = 0
        for n, m in self.named_modules():
            if isinstance(m, nn.Module) and \
                    not (n.endswith('fuse_layers.relu')):
                # if isinstance(m, nn.Conv2d):
                # n.replace('.', '_')
                heatmap_dir = os.path.join('HeatMaps', self._get_name(), 'forward', '@', str(global_n) + n + '.png')
                m.register_forward_hook(print_heatmap(heatmap_dir, 'forward'))
                heatmap_dir = os.path.join('HeatMaps', self._get_name(), 'backward', '@', str(global_n) + n + '.png')
                m.register_backward_hook(print_heatmap(heatmap_dir, 'backward'))
                fftmap_dir = os.path.join('FftMaps', self._get_name(), 'forward', '@', str(global_n) + n + '.png')
                m.register_forward_hook(print_fftmap(fftmap_dir, 'forward'))
                fftmap_dir = os.path.join('FftMaps', self._get_name(), 'backward', '@', str(global_n) + n + '.png')
                m.register_backward_hook(print_fftmap(fftmap_dir, 'backward'))
                global_n += 1

    def clean(self, ):
        del self.backbone
        del self.class_net
        del self.box_net
        torch.cuda.empty_cache()


def get_pose_net(cfg, is_train, is_registered=False, **kwargs):
    model_name = cfg.MODEL.BACKBONE
    checkpoint_path = cfg.MODEL.PRETRAINED
    model = EffPNet(
        get_efficientdet_config(model_name),
        pretrained_backbone=is_train,
        cfg=cfg,
    )
    if is_train and cfg.MODEL.INIT_WEIGHTS and os.path.isfile(checkpoint_path):
        model.load_ckpt(checkpoint_path)
    if is_registered:
        model.visualization()
    model.clean()
    model.train()
    return model


if __name__ == '__main__':
    import time
    import pprint
    import torchvision.transforms as transforms
    from ptflops import get_model_complexity_info as get_mc_info
    from thop import profile, clever_format
    from tools import _init_paths
    from config import cfg
    from config import update_config
    from utils.utils import parse_args
    from utils.utils import create_logger
    from utils.utils import get_model_summary
    import dataset

    # Update config
    global args
    args = parse_args()
    args.world_size = 1
    update_config(cfg, args)
    # Creat logger
    logger, final_output_dir, _ = create_logger(cfg, args.cfg, 'model_test')
    logger.info(args)
    logger.info(pprint.pformat(cfg))
    with torch.no_grad():
        n = 100
        is_train = False
        is_registered = False
        cfg.defrost()
        cfg.freeze()
        dump_input = torch.zeros((1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])).cuda()

        # model summary by thop
        model = get_pose_net(cfg, is_train, is_registered).cuda()
        # FP16
        if cfg.FP16.ENABLED:
            from fp16_utils.fp16util import network_to_half
            model = network_to_half(model)
        macs2, params2 = profile(model, inputs=(dump_input[0:1],), verbose=False)
        macs2, params2 = clever_format([macs2, params2], "%.3f")
        logger.info('#' * 96)
        logger.info('model summary by thop')
        logger.info('{:<30}  {:<8}'.format('Computational complexity: ', macs2))
        logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params2))

        # details for model summary by ptflops
        model = get_pose_net(cfg, is_train, is_registered).cuda()
        # FP16
        if cfg.FP16.ENABLED:
            from fp16_utils.fp16util import network_to_half
            model = network_to_half(model)
        with open(logger.handlers[0].baseFilename, 'a+') as f:
            logger.info('#' * 96)
            logger.info('details for model summary by ptflops')
            macs1, params1 = get_mc_info(model, tuple(dump_input.shape[1:4]),
                                         as_strings=True, print_per_layer_stat=True, ost=f)
        os.rename(logger.handlers[0].baseFilename, logger.handlers[0].baseFilename.replace('.log', '.sql'))

    # # Data loading code
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # dataset = eval('dataset.' + cfg.DATASET.DATASET)(cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
    #                                                  transforms.Compose([transforms.ToTensor(), normalize, ]))
    # loader = torch.utils.data.DataLoader(dataset, batch_size=1,
    #                                      shuffle=False, num_workers=cfg.WORKERS, pin_memory=True)
    # if cfg.TEST.MODEL_FILE:
    #     checkpoint_file = cfg.TEST.MODEL_FILE
    # else:
    #     checkpoint_file = ''
    # if os.path.exists(checkpoint_file):
    #     is_train = False
    #     is_registered = True
    #     model = get_pose_net(cfg, is_train, is_registered).cuda()
    #     # FP16
    #     if cfg.FP16.ENABLED:
    #         from fp16_utils.fp16util import network_to_half
    #
    #         model = network_to_half(model)
    #     model.eval()
    #     checkpoint = torch.load(checkpoint_file, map_location='cpu')
    #     ret = model.load_state_dict(checkpoint, strict=False)
    #     logger.info(ret)
    #     logger.info('==>Model loaded from ' + checkpoint_file)
    #     # Inferences
    #     torch.cuda.synchronize()
    #     begin_time = time.time()
    #     for i, (image, target, target_weight, meta) in enumerate(loader):
    #         # compute output
    #         image = image.cuda(non_blocking=True)
    #         target = target.cuda(non_blocking=True)
    #         outputs = model(image)
    #
    #         import cv2
    #
    #         mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    #         std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
    #         cv2.imshow(
    #             'input[0]',
    #             cv2.cvtColor(
    #                 (
    #                         image * std + mean + torch.nn.Upsample(scale_factor=4, mode='nearest')(
    #                     outputs[-1].sum(1, keepdim=True)).permute(0, 2, 3, 1)[0].detach().cpu().numpy()
    #                 ),
    #                 cv2.COLOR_BGR2RGB
    #             )
    #         )
    #         cv2.waitKey()
    #         # plt.imshow(
    #         #     (0. * torch.nn.Upsample(scale_factor=4, mode='nearest')(outputs[-1].sum(1, keepdim=True))
    #         #      + 1. * image).permute(0, 2, 3, 1)[0].detach().cpu().numpy())
    #         # plt.show()
    #         torch.mean((outputs[-1] - target) ** 2).backward()
    #         break
    #     torch.cuda.synchronize()
    #     end_time = time.time()
    #     logger.info('Inferences with feature visualization: ' + str(end_time - begin_time) + 's spent.')
