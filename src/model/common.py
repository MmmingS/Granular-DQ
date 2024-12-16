import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from decimal import Decimal, ROUND_HALF_UP

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ShortCut(nn.Module):
    def __init__(self):
        super(ShortCut, self).__init__()

    def forward(self, input):
        return input

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std).cuda()
        self.weight.data = torch.eye(3).view(3, 3, 1, 1).cuda() / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean).cuda() / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, inn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            elif inn:
                m.append(nn.InstanceNorm2d(n_feats, affine=True))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        self.shortcut = ShortCut()

    def forward(self, x):
        residual = self.shortcut(x)
        # print("hi")
        res = self.body(x).mul(self.res_scale)
        res += residual

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias=bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))
                elif act == 'lrelu':
                    m.append(nn.LeakyReLU(0.2, inplace=True))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
            elif act == 'lrelu':
                    m.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Upsampler_q(nn.Module):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True, k_bits=32, ema_epoch=1, search_space=[4,6,8]):
        super(Upsampler_q, self).__init__()
        
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                # m.append(quant_act_pams(k_bits, ema_epoch=ema_epoch))
                m.append(classify(n_feats, bias=bias, ema_epoch=ema_epoch, search_space=search_space))
                m.append(conv(n_feats, 4 * n_feats, kernel_size=3, bias=bias, k_bits = k_bits))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        self.m = nn.Sequential(*m)

    def forward(self,x):
        weighted_bits = x[3]
        bits = x[2]
        grad = x[0]
        x = x[1]

        # check if the pretrained model's name is like this
        # return self.m(x)
        grad, x, bits, weighted_bits = self.m[0]([grad,x,bits,weighted_bits])
        x= self.m[1:3](x)
        grad, x, bits, weighted_bits = self.m[3]([grad,x,bits,weighted_bits])
        x= self.m[4:6](x)

        return [grad, x, bits, weighted_bits]
        
class ResBlock_srresnet(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=False, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock_srresnet, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)

        self.bn1 = nn.BatchNorm2d(n_feats)
        self.act = act
        self.bn2 = nn.BatchNorm2d(n_feats)
        self.res_scale = res_scale


        self.res_scale = res_scale
        self.shortcut = ShortCut()

    def forward(self, x):
        residual = self.shortcut(x)
        res = self.act(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res)).mul(self.res_scale)
        res += residual
        # residual = self.shortcut(x)
        # res = self.body(x).mul(self.res_scale)
        # res += residual
        return res

class Upsampler_srresnet(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        # scale = 4 # for SRResNet
        m = []
        if scale == 4:
            m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
            m.append(nn.PixelShuffle(2))
            m.append(nn.PReLU())
            # m.append(nn.LeakyReLU(0.2, inplace=True))
            m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
            m.append(nn.PixelShuffle(2))
            # m.append(nn.LeakyReLU(0.2, inplace=True))
            m.append(nn.PReLU())
        elif scale ==2 :
            m.append(conv(n_feats, 4 * n_feats, 3, bias=False))
            m.append(nn.PixelShuffle(2))
            m.append(nn.PReLU())
        else:
            print("not implemented")
        

        super(Upsampler_srresnet, self).__init__(*m)
        
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)
    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()
        
    def entropy(self, values, bins, sigma, batch):
        epsilon = 1e-40
        values = values / 255.0
        values = values.unsqueeze(2)
        sigma = sigma * 255.0
        residuals = values - bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))
        kernel_values[kernel_values < epsilon] = epsilon
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
        pdf = pdf / normalization + epsilon
        entropy = -torch.sum(pdf * torch.log(pdf), dim=1)
        entropy = entropy.reshape((batch, -1))
        entropy = rearrange(entropy, "B (H W) -> B H W", H=self.wh, W=self.hw)
        return entropy

    def forward(self, inputs):
        if inputs.dim() == 3:  # Handling three-dimensional inputs [C, H, W]
            inputs = inputs.unsqueeze(0)  # Convert to four-dimensional [1, C, H, W]
        self.width = inputs.size(2)
        self.height = inputs.size(3)
        self.psize = min(self.width, self.height)
        self.patch_num = int(self.width * self.height / self.psize ** 2)
        self.hw = int(self.width // self.psize)
        self.wh = int(self.height // self.psize)
        unfold = torch.nn.Unfold(kernel_size=(self.psize, self.psize), stride=self.psize)
        batch_size = inputs.shape[0]
        gray_images = 0.2989 * inputs[:, 0:1, :, :] + 0.5870 * inputs[:, 1:2, :, :] + 0.1140 * inputs[:, 2:, :, :]
        unfolded_images = unfold(gray_images)
        unfolded_images = unfolded_images.transpose(1, 2)
        unfolded_images = torch.reshape(unfolded_images.unsqueeze(2),
                                        (unfolded_images.shape[0] * self.hw * self.wh, unfolded_images.shape[2]))
        entropy = self.entropy(unfolded_images, bins=torch.linspace(0, 1, 32).to(inputs.device),
                               sigma=torch.tensor(0.0001), batch=batch_size)
        return entropy

class Encoder(nn.Module):
    def __init__(self, 
    *, 
    num_channels=64,
    **ignore_kwargs
    ):
        super().__init__()
        self.num_splits = 3
        self.down_1 = Downsample(in_channels=64, with_conv=False)
        self.down_2 = Downsample(in_channels=64, with_conv=False)
        self.gate_median_pool = nn.AvgPool2d(2, 2)
        self.avp1 = nn.AdaptiveAvgPool2d(1)
        self.avp2 = nn.AdaptiveAvgPool2d(1)
        self.avp3 = nn.AdaptiveAvgPool2d(1)
        self.gate_fine_pool = nn.AvgPool2d(4, 4)
        # self.feature_norm_fine = nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-6, affine=True)
        # self.feature_norm_median = nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-6, affine=True)
        # self.feature_norm_coarse = nn.GroupNorm(num_groups=8, num_channels=64, eps=1e-6, affine=True)
        self.gate = nn.Sequential(
            nn.Linear(num_channels * self.num_splits, 3)#(64*3,3)
        )
        self.calculate_entropy = Entropy()
        self.calculate_entropy = self.calculate_entropy.eval()
        self.calculate_entropy.train = disabled_train
        self.second_divide_bit = TripleGrainEntropyBitControler()
        
    def forward(self, x, image, threshold):
        x_fine = x 
        x = self.down_1(x_fine) 
        x_median = x 
        x = self.down_2(x_median)
        x_coarse = x 
        avg_x_fine = self.gate_fine_pool(x_fine) 
        avg_x_median = self.gate_median_pool(x_median)
        x_coarse_cat = self.avp1(x_coarse) 
        avg_x_fine_cat = self.avp2(avg_x_fine) 
        avg_x_median_cat = self.avp3(avg_x_median)
        x_logistic = torch.cat([x_coarse_cat, avg_x_fine_cat, avg_x_median_cat], dim=1).permute(0,2,3,1)
        gate = self.gate(x_logistic)
        gate = F.gumbel_softmax(gate, tau=1, dim=-1, hard=True)
        gate = gate.permute(0,3,1,2)
        if self.training:
            indices = gate.argmax(dim=1).view(16)
        else:
            indices = gate.argmax(dim=1).squeeze().view(1)
        indices_repeat = indices.to('cpu')
        indices_repeat = torch.where(indices_repeat == 2, torch.tensor(8), indices_repeat)
        indices_repeat = torch.where(indices_repeat == 1, torch.tensor(6), indices_repeat) 
        indices_repeat = torch.where(indices_repeat == 0, torch.tensor(4), indices_repeat).to('cuda:0')
        for i,bit in enumerate(indices_repeat):
            if bit == 8:
                image_detached = image[i].detach()
                entropy = self.calculate_entropy(image_detached).detach()
                indices_repeat[i] = self.second_divide_bit(entropy)
        return indices_repeat

class TripleGrainEntropyBitControler(nn.Module):
    def __init__(self,
                 json_path = "/home/wms/Granular-DQ/src/scripts/E2B.json", 
                 decay=0.99997,
                 ):
        super().__init__()
        self.decay = decay
        self.fine_grain_ratito1 = nn.Parameter(torch.tensor([0.50]), requires_grad=False)
        self.fine_grain_ratito2 = nn.Parameter(torch.tensor([0.90]), requires_grad=False)
        self.norm_value = 2.2
        with open(json_path, "r", encoding="utf-8") as f:
            self.content = json.load(f)
    
    def _ema(self,current_entropy):
        current_entropy = (current_entropy - 1.2) / self.norm_value
        # print('x:'+str(current_entropy))
        self.fine_grain_ratito1.data = self.fine_grain_ratito1.data*self.decay + (1-self.decay)*current_entropy
        self.fine_grain_ratito2.data = self.fine_grain_ratito2.data*self.decay + (1-self.decay)*current_entropy
        # print(self.fine_grain_ratito1)
        # print(self.fine_grain_ratito2)
        
    def round_to_nearest_half(self,value):
        decimal_value = Decimal(str(value))
        rounded_value = decimal_value.quantize(Decimal('0.5'), rounding=ROUND_HALF_UP)
        rounded_value = max(min(rounded_value, 1), 0.05)
        return rounded_value
    
    def forward(self,entropy,threshold=None):
        if self.training and self.epoch <= 1:
            self._ema(entropy)
        fine_grain_threshold1 = self.content["{:.1f}".format(self.round_to_nearest_half(self.fine_grain_ratito1.item()) * 100)]
        fine_grain_threshold2 = self.content["{:.1f}".format(self.round_to_nearest_half(self.fine_grain_ratito2.item()) * 100)]
        if entropy > fine_grain_threshold2: gate = 8
        elif entropy <= fine_grain_threshold1: gate = 4
        else: gate = 5
        return gate    

