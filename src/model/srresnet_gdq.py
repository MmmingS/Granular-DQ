import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia as K
from model.common import Encoder,MeanShift,Upsampler_srresnet
from model.quant_ops import conv3x3,default_init_weights,quant_conv3x3_quantsr,conv9x9,quant_act_quantsr

class QResBlockWithBN(nn.Module):
    """        
           ---Conv-ReLU-Conv-+-
            |________________|
    """
    def __init__(self, num_feat=64, res_scale=1,pytorch_init=False, bias=False):
        super(QResBlockWithBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = quant_conv3x3_quantsr(num_feat, num_feat, 3, 1, 1, bias=False, k_bits=8) #优化bits
        self.conv2 = quant_conv3x3_quantsr(num_feat, num_feat, 3, 1, 1, bias=False, k_bits=8)
        self.bn1 = nn.BatchNorm2d(num_feat)
        self.act = nn.PReLU()
        self.act_q = quant_act_quantsr(k_bits=8)
        self.bn2 = nn.BatchNorm2d(num_feat)

        self.learnable_shortcut = nn.Parameter(torch.ones(1), requires_grad=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2,self.bn1,self.bn2], 0.1)
        
    def forward(self,x):
        f = x[2]
        bits_out = x[3]
        bits = x[1]
        identity = x[0]
        
        x,bits,bits_out = self.conv1([identity,bits,bits_out])
        x = self.bn1(self.act(x))
        out,bits,bits_out = self.conv2([x,bits,bits_out])
        out = self.bn2(out)
        out = identity * self.learnable_shortcut + (out * self.res_scale)
        
        return [out,bits,f,bits_out]

class SRResNet_GDQ(nn.Module):
    def __init__(self, args, conv=quant_conv3x3_quantsr, bias=False, k_bits=32):
        super(SRResNet_GDQ, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        scale = args.scale[0]
        act = nn.PReLU()
        # act = nn.LeakyReLU(0.2, inplace=True)
        self.k_bits = k_bits
        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)
        m_head = [conv9x9(args.n_colors, n_feats, kernel_size=9, bias=False)]
        m_head.append(act)
        self.encoder = Encoder()
        m_body = [
            QResBlockWithBN(n_feats, res_scale=args.res_scale, pytorch_init=True,bias=bias
                ) for _ in range(n_resblocks)
        ]
        m_body.append(conv3x3(n_feats, n_feats, kernel_size, bias=False))
        m_body.append(nn.BatchNorm2d(n_feats))
        
        m_tail = [
            Upsampler_srresnet(conv3x3, scale, n_feats, act=False),
            conv9x9(n_feats, args.n_colors, kernel_size=9, bias=False)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, threshold):
        
        f=None; bits_out = torch.zeros(x.size(0)).to(x.device)
        image = x
        x = self.head[0:1](x)
        bit_index = self.encoder(x,image,threshold)
        x = self.head[1:](x)
        res = x
        res, bits, f,bits_out = self.body[0:-2]([res, bit_index, f, bits_out])
        res = self.body[-2:](res)
        res += x
        out = res
        x = self.tail(res)
        
        return x, out, bits, f, bits_out


