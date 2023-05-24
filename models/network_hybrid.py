# 测试
# 开发时间：2022/8/5 9:26
from thop import profile
# import torchvision.models as models
# import torch
from ptflops import get_model_complexity_info
from models.basicblock import DRB, PALayer, CALayer, CCALayer, SRB
from models.SwinT import SwinT
# from .FCVit import fcvit_block
# from .FCA import MultiSpectralAttentionLayer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as trans_fn
from torchvision.transforms import InterpolationMode
from models.fusion import iAFF ,AFF, MS_CAM

# 双三次上采样
# img = trans_fn.resize(img, size, InterpolationMode.BICUBIC)

def channel_shuffle(x, groups=4):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups# reshape
    x = x.view(batchsize, groups,channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class myModel(nn.Module):
    def __init__(self, img_size=64, num_heads=8, upscale=4, window_size=8, num_in_ch=3, nf=64, embed_dim=64,
                 depth=4, upsampler='pixelshuffledirect', img_range=1.):
        super(myModel, self).__init__()
        num_feat = 64
        num_out_ch = 3
        self.upsampler = upsampler
        self.window_size = window_size
        self.img_range = img_range
        self.upscale = upscale
        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = depth
        self.layers = nn.ModuleList()  # 存放HRBCT模块
        for i_layer in range(self.num_layers):
            layer = HRBCT(embed_dim, nf, num_heads)
            self.layers.append(layer)

        #####################################################################################################
        ################################### 2.2, 深度特征融合模块 ######################################
        self.conv1 = nn.Conv2d(depth*embed_dim, embed_dim, kernel_size=1) # depth*embed_dim
        self.conv3 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=True)
        self.PA = PALayer(embed_dim) #

        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (img_size, img_size))

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_shallow_features(self, x):
        x1 = self.RRDB(x)
        x1 = self.firstUp(x1)
        x1 = self.conv_end1(x1)
        return x1

    def forward_features(self, x):

        retainV = []
        for layer in self.layers:
            x = layer(x)
            retainV.append(x)

        x1 = torch.cat((retainV[0], retainV[1], retainV[2], retainV[3]), 1).contiguous()

        return x1

    def DFF(self, x):
        x1 = self.conv1(x)
        x1 = self.conv3(x1)
        x1 = self.PA(x1)
        return x1

    def forward(self, x):
        H, W = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':

            x = self.conv_first(x)
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':

            x = self.conv_first(x)  # 经过浅层特征提取

            x = self.DFF(self.forward_features(x)) + x  # 经过深层特征提取和特征融合

            x = self.upsample(x)  # 图像上采样重建

        x = x / self.img_range + self.mean

        return x[:, :, :H * self.upscale, :W * self.upscale]


class HRBCT(nn.Module):
    def __init__(self, embed_dim=64, nf=64, num_heads=8,distillation_rate=0.50):
        super(HRBCT, self).__init__()
        # 知识蒸馏
        self.distilled_channels = int(embed_dim * distillation_rate)
        self.remaining_channels = int(embed_dim - self.distilled_channels)
        self.distillation_rate = distillation_rate

        self.Conv3_D1 = nn.Conv2d(self.distilled_channels, self.distilled_channels, 3, 1, 1)
        self.Conv3_D2 = nn.Conv2d(int(self.remaining_channels * self.distillation_rate), int(self.remaining_channels * self.distillation_rate), 3, 1, 1)

        self.ST = SwinT(embed_dim=self.remaining_channels, heads=num_heads)



        self.SRB = SRB(int(nf*(1-distillation_rate)**2))

        # self.BSRB = BSConvURB( int(nf*(1-distillation_rate)**2), int(nf*(1-distillation_rate)**2), kernel_size=3)

        # DRB
        # self.DRB = DRB(int(nf*(1-distillation_rate)**2))

        # ESA
        # self.ESA = ESA(n_feats=nf, conv=nn.Conv2d)  # 输出通道 输入通道

        self.CCA = CCALayer(nf)

    def forward(self, x):

        distilled_c1, remaining_c1 = torch.split(x, (self.distilled_channels, self.remaining_channels), dim=1)


        distilled_c1 = self.Conv3_D1(distilled_c1)

        out1 = self.ST(remaining_c1)



        distilled_c2, remaining_c2 = torch.split(out1, (int(self.remaining_channels*self.distillation_rate), int(self.remaining_channels*(1-self.distillation_rate))), dim=1)

        distilled_c2 = self.conv1_D2(distilled_c2)

        # distilled_c2 = self.Conv3_D2(distilled_c2)
        #
        out2 = self.SRB(remaining_c2)

        out = torch.cat([distilled_c1, distilled_c2, out2], dim=1)
        x1 = self.CCA(out) #

        x_4 = x + x1

        return x_4



if __name__ == '__main__':
    x = torch.randn((1, 3, 64, 64))
    model = myModel()
    y = model(x)
    print(y.shape)
    device = torch.device('cuda:0')
    input = x.to(device)
    model.eval()
    model = model.to(device)
    macs, params = get_model_complexity_info(model, (3, 64, 64), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
