import torch.nn as nn
import torch
import copy
import math

from torch.nn import functional as F
from torch.nn import SiLU as MemoryEfficientSwish
from torch.nn import SiLU as Swish
from .utils_extra import Conv2dStaticSamePadding
from timm.models.efficientnet_blocks import SqueezeExcite, ConvBnAct, \
    DepthwiseSeparableConv, InvertedResidual, CondConvResidual, EdgeResidual
from timm.models.layers import create_conv2d, drop_path

# activation = True
activation = False
# attention = True
attention = False


class Identity(nn.Module):
    def __init__(self, in_channel, out_channel, drop_path_rate=0., has_bn=True):
        super(Identity, self).__init__()
        if in_channel != out_channel:
            if has_bn:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, bias=False),
                    nn.BatchNorm2d(out_channel, momentum=0.01, eps=1e-3)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel, 1, bias=True),
                )
        else:
            self.conv = None
        # self.drop_path_rate = drop_path_rate

    def forward(self, x):
        if self.conv is None:
            # if self.drop_path_rate > 0.:
            #     x = drop_path(x, self.drop_path_rate, self.training)
            return x
        else:
            return self.conv(x)


class WeightedAdd(nn.Module):
    def __init__(self, in_channels, out_channel, out_rank, attention=False,
                 activation=True, onnx_export=False, epsilon=1e-5):
        super(WeightedAdd, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channel, int)

        self.sample_layers = nn.ModuleList()
        for in_rank in range(len(in_channels)):
            'the rank means the index of feature maps, which is zero when the resolution is the largest.'
            if in_rank > out_rank:
                self.sample_layers.append(
                    nn.Sequential(
                        Identity(in_channels[in_rank], out_channel),
                        # nn.Upsample(scale_factor=2 ** (in_rank - out_rank), mode='bilinear', align_corners=False),
                        nn.Upsample(scale_factor=2 ** (in_rank - out_rank), mode='bilinear', align_corners=True),
                        # nn.Upsample(scale_factor=2 ** (in_rank - out_rank), mode='nearest'),
                    )
                )
            elif in_rank == out_rank:
                self.sample_layers.append(
                    nn.Sequential(
                        Identity(in_channels[in_rank], out_channel),
                    )
                )
            else:
                self.sample_layers.append(
                    nn.Sequential(
                        nn.MaxPool2d(2 ** (out_rank - in_rank), 2 ** (out_rank - in_rank)),
                        # nn.Upsample(scale_factor=2 ** (in_rank - out_rank), mode='bilinear', align_corners=True),
                        Identity(in_channels[in_rank], out_channel),
                    )
                )

        self.attention = attention
        if self.attention:
            se_ratio = 0.25
            num_squeezed_channels = max(1, int(sum(in_channels) * se_ratio))
            self._se_reduce = nn.Conv2d(sum(in_channels), num_squeezed_channels, 1)
            self._swish = MemoryEfficientSwish() if not onnx_export else Swish()
            self._se_expand = nn.Conv2d(num_squeezed_channels, out_channel * len(in_channels), 1)
        else:
            self.relu = nn.ReLU()
            self.epsilon = epsilon
            self.weights = nn.Parameter(
                (1. / len(in_channels)) * torch.ones((1, out_channel, len(in_channels), 1, 1), dtype=torch.float32),
                requires_grad=True,
            )

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, inputs):
        assert isinstance(inputs, list)
        if self.attention:
            x_squeezed = torch.cat([F.adaptive_avg_pool2d(input, 1) for input in inputs], dim=1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            weight = torch.sigmoid(x_squeezed)
            weight = weight.reshape([weight.shape[0], -1, len(inputs), weight.shape[2], weight.shape[3]])
        else:
            weight = self.relu(self.weights)
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + self.epsilon)

        output = '0.0'
        for j in range(len(inputs)):
            output += ' + weight[:, :, {0}, :, :]*self.sample_layers[{0}](inputs[{0}])'.format(j)

        if self.activation:
            return self.swish(eval(output))
        else:
            return eval(output)


class TotalFusion(nn.Module):
    def __init__(self, in_channels, out_channels, drop_path_rates=[0.], onnx_export=False):
        super(TotalFusion, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, list)
        assert len(in_channels) == len(out_channels) == len(drop_path_rates)

        self.addup_layers = nn.ModuleList(
            WeightedAdd(in_channels, out_channel, out_rank, attention=False, activation=True, onnx_export=onnx_export)
            for out_rank, out_channel in enumerate(out_channels)
        )

        self.mixup_layers = nn.ModuleList(
            MBBlock(
                out_channel, out_channel, 5, noskip=True, exp_ratio=6.0,
                act_layer=MemoryEfficientSwish if not onnx_export else Swish,
                se_ratio=0.25, se_layer=SqueezeExcite, drop_path_rate=drop_path_rate,
            )
            for out_channel, drop_path_rate in zip(out_channels, drop_path_rates)
        )

        self.fixup_layers = nn.ModuleList(
            Identity(in_channel, out_channel)
            for in_channel, out_channel in zip(in_channels, out_channels)
        )

    def forward(self, inputs):
        outputs = []
        for addup_layer, mixup_layer, fixup_layer, input in zip(
                self.addup_layers, self.mixup_layers, self.fixup_layers, inputs
        ):
            outputs.append(
                mixup_layer(addup_layer(inputs)) + fixup_layer(input)
            )
        return outputs


class FinalAttentionHead(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(FinalAttentionHead, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channel, int)

        scale = len(in_channels)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels[i], out_channels=out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_features=out_channel, momentum=0.01, eps=1e-3),
                nn.Sigmoid(),
                nn.Upsample(scale_factor=2 ** i, mode='nearest'),
            )
            # CondOutLayer(in_channels[i], out_channel, scale_factor=2**i)
            for i in range(scale)
        )

    def forward(self, inputs):
        out = '1.0'
        for i in range(len(inputs)):
            out += '*self.layers[{i}](inputs[{i}])'.format(i=i)
        return eval(out)


class FinalSumHead(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(FinalSumHead, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channel, int)

        scale = len(in_channels)
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels[i], out_channels=out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_features=out_channel, momentum=0.01, eps=1e-3),
                nn.Upsample(scale_factor=2 ** i, mode='nearest'),
            )
            for i in range(scale)
        )

    def forward(self, inputs):
        out = '0.0'
        for i in range(len(inputs)):
            out += '+self.layers[{i}](inputs[{i}])'.format(i=i)
        return eval(out)


class MBBlock(InvertedResidual):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def forward(self, x):
        shortcut = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.drop_path_rate > 0.:
            x = drop_path(x, self.drop_path_rate, self.training)
        if self.has_residual:
            x += shortcut

        return x


class HighResolutionBlock(nn.Module):
    def __init__(self, hr_branches, new_block, recurrent=False, need_fuse=False, onnx_export=False):
        super(HighResolutionBlock, self).__init__()
        self.branches = nn.ModuleList()
        for hr_branch in hr_branches:
            if recurrent:
                # Parameter shared, recurrent cnn
                block = len(new_block) * [hr_branch]
            else:
                # Parameter not shared, normal cnn
                block = nn.ModuleList()
                for new_b in new_block:
                    b = copy.deepcopy(hr_branch)
                    b.drop_path_rate = new_b.drop_path_rate
                    block.append(b)
            self.branches.append(nn.Sequential(*block))
        self.branches.append(new_block)
        self.drop_path_rates = len(self.branches) * [self.branches[-1][-1].drop_path_rate]

        self.onnx_export = onnx_export
        self.fuse_layers = None
        if need_fuse:
            self.make_fuse_layer()

    def make_fuse_layer(self):
        channels = []
        for branch in self.branches:
            channels.append(branch[-1].feature_info('bottleneck')['num_chs'])

        if len(channels) > 1:
            self.fuse_layers = TotalFusion(channels, channels, drop_path_rates=self.drop_path_rates)
        else:
            self.fuse_layers = None

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        num = len(self.branches) - len(inputs)
        assert (num == 0 or num == 1)
        outputs = []
        for input, branch in zip(inputs, self.branches):
            outputs.append(branch(input))
        if num == 1:
            outputs.append(self.branches[-1](inputs[-1]))

        if self.fuse_layers is not None:
            outputs = self.fuse_layers(outputs)
        return outputs


def _init_weight(m, n='', ):
    """ Weight initialization as per Tensorflow official implementations.
    """

    def _fan_in_out(w, groups=1):
        dimensions = w.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        num_input_fmaps = w.size(1)
        num_output_fmaps = w.size(0)
        receptive_field_size = 1
        if w.dim() > 2:
            receptive_field_size = w[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
        fan_out //= groups
        return fan_in, fan_out

    def _glorot_uniform(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., (fan_in + fan_out) / 2.)  # fan avg
        limit = math.sqrt(3.0 * gain)
        w.data.uniform_(-limit, limit)

    def _variance_scaling(w, gain=1, groups=1):
        fan_in, fan_out = _fan_in_out(w, groups)
        gain /= max(1., fan_in)  # fan in
        # gain /= max(1., (fan_in + fan_out) / 2.)  # fan

        # should it be normal or trunc normal? using normal for now since no good trunc in PT
        # constant taken from scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # std = math.sqrt(gain) / .87962566103423978
        # w.data.trunc_normal(std=std)
        std = math.sqrt(gain)
        w.data.normal_(std=std)

    if isinstance(m, InvertedResidual):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            if m.conv_dw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_dw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_dw.bias.data.zero_()
            _variance_scaling(m.conv_pwl.weight)
            if m.conv_pwl.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pwl.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pwl.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            if m.conv_dw.bias is not None:
                m.conv_dw.bias.data.zero_()
            _glorot_uniform(m.conv_pwl.weight)
            if m.conv_pwl.bias is not None:
                m.conv_pwl.bias.data.zero_()
        # print(f'_init_weight: {n} by {type(m)}')
    elif isinstance(m, DepthwiseSeparableConv):
        if 'box_net' in n or 'class_net' in n:
            _variance_scaling(m.conv_dw.weight, groups=m.conv_dw.groups)
            if m.conv_dw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_dw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_dw.bias.data.zero_()
            _variance_scaling(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_pw.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_pw.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_dw.weight, groups=m.conv_dw.groups)
            if m.conv_dw.bias is not None:
                m.conv_dw.bias.data.zero_()
            _glorot_uniform(m.conv_pw.weight)
            if m.conv_pw.bias is not None:
                m.conv_pw.bias.data.zero_()
        # print(f'_init_weight: {n} by {type(m)}')
    elif isinstance(m, SqueezeExcite):
        if 'box_net' in n or 'class_net' in n:
            m.conv_reduce.weight.data.normal_(std=.01)
            if m.conv_reduce.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_reduce.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_reduce.bias.data.zero_()
            m.conv_expand.weight.data.normal_(std=.01)
            if m.conv_expand.bias is not None:
                if 'class_net.predict' in n:
                    m.conv_expand.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
                else:
                    m.conv_expand.bias.data.zero_()
        else:
            _glorot_uniform(m.conv_reduce.weight)
            if m.conv_reduce.bias is not None:
                m.conv_reduce.bias.data.zero_()
            _glorot_uniform(m.conv_expand.weight)
            if m.conv_expand.bias is not None:
                m.conv_expand.bias.data.zero_()
        # print(f'_init_weight: {n} by {type(m)}')
    elif isinstance(m, nn.BatchNorm2d):
        # looks like all bn init the same?
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
        # print(f'_init_weight: {n} by {type(m)}')


def _init_weight_alt(m, n='', ):
    """ Weight initialization alternative, based on EfficientNet bacbkone init w/ class bias addition
    NOTE: this will likely be removed after some experimentation
    """
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            if 'class_net.predict' in n:
                m.bias.data.fill_(-math.log((1 - 0.01) / 0.01))
            else:
                m.bias.data.zero_()
        # print(f'_init_weight_alt: {n} by {type(m)}')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
        # print(f'_init_weight_alt: {n} by {type(m)}')
