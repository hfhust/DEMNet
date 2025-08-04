import torch
import torch.nn as nn
from models.layers import DTUM
from models.layers import DTUM_concat
import torch.nn.functional as F
from functools import partial

def _upsample_like(src, tar):
    #print('str和tar的形状为:'+str(src.shape)+str(tar.shape))
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class last_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
         #   DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up_fuse(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_mid = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv_mid = DoubleConv(in_channels, out_channels)

        self.fuse = fuse(out_channels)
        self.conv = DoubleConv(out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_mid(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = self.fuse(x1, x2)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.fuse = fuse(in_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = self.fuse(x1, x2)
        return self.conv(x)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return  x_LL, torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


class DWT_d4(nn.Module):
    def __init__(self):
        super(DWT_d4, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init_d4(x)

def dwt_init_d4(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    # print([in_batch, in_channel, in_height, in_width])
    device = x.device
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    h = torch.zeros([out_batch, out_channel, out_height, out_width], dtype=torch.float, device=device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class last_dwt1(nn.Module):
    def __init__(self, in_channels, out_channels, se):
        super().__init__()
        self.dwt = DWT_d4()
        # if se:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #         nn.BatchNorm2d(out_channels),
        #         SEModule(out_channels),
        #         nn.ReLU(inplace=True)
        #     )
        # else:
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_dwt = self.dwt(x)
       # out = self.conv(x_ll)

        return x_dwt

class part_dwt1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_ll, x_dwt = self.dwt(x)
        out = self.conv(x_ll)

        return out, x_dwt

class part_dwt2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_ll, x_dwt = self.dwt(x)
        out = self.conv(x_dwt)

        return out

class part_dwt3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_ll, x_dwt = self.dwt(x)
        out = self.conv(x_dwt)

        return out


class part_iwt1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_mid = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(out_channels, out_channels)
        self.iwt = IWT()
        self.fuse = fuse(out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_mid(x1)
        x2 = self.iwt(x2)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x1, x2)
        return self.conv(x)

class part_iwt1_fuse(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear):
        super().__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.up = nn.Upsample(size=48, mode='bilinear', align_corners=True)
            self.conv_mid = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(out_channels , out_channels)
        self.iwt = IWT()
        self.fuse = fuse(out_channels)

    def forward(self, x1, x2):
        # print('输入x1的形状是:'+str(x1.shape))

        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_mid(x1)

        
        x2 = self.iwt(x2)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        # x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x1, x2)

        return self.conv(x)

class part_iwt2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)
        self.iwt = IWT()

    def forward(self, x1, x2):
        x1 = self.iwt(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class part_iwt3(nn.Module):
    def __init__(self, mid_in, mid_out, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up =IWT()
        self.conv_mid = nn.Conv2d(mid_in, mid_out, kernel_size=1)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv_mid(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, size, interpolate_mode):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.interpolate = partial(F.interpolate,
                                   size=size,
                                   mode=interpolate_mode,
                                   align_corners=False,
                                   recompute_scale_factor=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.interpolate(x)
        return x

class out_conv(nn.Module):
    def __init__(self, ch1, ch2, n_classes):
        super().__init__()
      #  self.mid_conv = nn.Conv2d(ch2, ch1, kernel_size=1)
        self.out_conv = nn.Conv2d(ch1+ch2, n_classes, 1)
      #  self.fuse = fuse(ch1)
        
    def forward(self, x1, x2):
            x2 = F.upsample(x2, size=x1.shape[2:], mode='bilinear')

            out = torch.cat((x1,x2), dim=1)
            out = self.out_conv(out)
            return out
    
class fuse(nn.Module):
    def __init__(self, channels=64, r=4):
        super(fuse,self).__init__()
        self.channels = channels
        self.mid_channels = int(channels // r)

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.bottomup = nn.Sequential(
            nn.Conv2d(channels, self.mid_channels, kernel_size=1),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(),
            nn.Conv2d(self.mid_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )


    def forward(self, xh, xl):

        topdown_wei = self.topdown(xh)

        bottomup_wei = self.bottomup(xl)

        xs = 2 * (xl * topdown_wei) + 2 * (xh * bottomup_wei)

        xs = self.post(xs)

        return xs

class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class ResBlock(nn.Module):
    def __init__(self, channel_size: int, negative_slope: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size),
            nn.LeakyReLU(negative_slope, inplace=True),
            nn.Conv2d(channel_size, channel_size, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_size)
        )

    def forward(self, x):
        return x + self.block(x)

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class RSF(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(RSF, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
class Resizer(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel =3
        head_channel = 32


        # 0.88
        # head_channel = 16

        # first
        modules_head = [
            nn.Conv2d(input_channel, head_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(head_channel),
            nn.ReLU(inplace=True)]

        self.head = nn.Sequential(*modules_head)
        self.modules_body1 = (RSF(32, 32, 3, 1, 64, True, nl='RE'))
        self.modules_body2 = (RSF(32, 32, 3, 1, 64, True, nl='RE'))
        self.modules_body3 = (RSF(32, 32, 3, 1, 64, True, nl='RE'))
        self.modules_body4 = (RSF(32, 32, 3, 1, 64, True, nl='RE'))
        modules_tail = []
        modules_tail.append(nn.Conv2d(32, 64, 1, padding=0, stride=1))
        # modules_tail.append(nn.Conv2d(96, 288, 1, padding=0, stride=1))
        modules_tail.append(nn.BatchNorm2d(64))
        modules_tail.append(nn.ReLU(True))
        modules_tail.append(nn.Conv2d(64, 3, 1, padding=0, stride=1))

        self.tail = nn.Sequential(*modules_tail)
        self.interpolate = partial(F.interpolate,
                                   size=512,
                                   mode='bilinear',
                                   align_corners=False,
                                   recompute_scale_factor=False)
    def forward(self, x):

        identity = x

        x = self.head(x)

        out1 = self.modules_body1(x)
        out2 = self.modules_body2(out1)
        out3 = self.modules_body3(out2)


        out = self.modules_body4(out3)


        out = self.tail(out)

        identity = self.interpolate(identity)

        return out + identity       



class WATP(nn.Module):
    def __init__(self,DTUM_True,size):
        super(WATP, self).__init__()
        self.n_classes = 1
        self.in_resizer = Resizer()

        self.layer = 4
        self.mode = 1
        self.n_channels = 3
        self.DTUM_True=DTUM_True
        self.size=size
        if self.mode == 1:
            if self.layer == 4:
                self.inc = DoubleConv(self.n_channels, 32)
                # self.inc = DoubleConv(1, 64)
                self.down1 = (part_dwt1(32, 64))
                self.down2 = (part_dwt1(64, 128))
                self.down3 = (part_dwt1(128, 256))
                self.down4 = (last_dwt1(256, 512, False))

                self.up4 = (part_iwt1_fuse(64, 256, True))

                self.conv4 = (out_conv(256, 128, 8))

                self.up3 = (part_iwt1(256, 128, True))

                self.conv3 = (out_conv(128, 64, 8))
                
                self.up2 = (part_iwt1(128, 64, True))

                self.conv2 = (out_conv(64, 32, 8))

                self.up1 = (part_iwt1(64, 32, True))
                self.conv1 = nn.Conv2d(32, 8, 1)
                if self.DTUM_True==1:
                    self.out = (OutConv(8 * self.layer, 16, self.size, 'bilinear'))
                else:
                    self.out = (OutConv(8 * self.layer, 1, self.size, 'bilinear'))

        

    def forward(self, x):
        x = self.inc(x)
        if self.mode == 1:
            if self.layer == 4:
                x1, x1_dwt = self.down1(x)
             
                x2, x2_dwt = self.down2(x1)

                x3, x3_dwt = self.down3(x2)
              
                x4_dwt = self.down4(x3)
                x4_up = self.up4(x1, x4_dwt)
                
                x3_up = self.up3(x4_up, x3_dwt)
                x2_up = self.up2(x3_up, x2_dwt)

                x1_up = self.up1(x2_up, x1_dwt)

                x1_out = self.conv1(x1_up)

                x2_out = self.conv2(x2_up, x1_up)
                x2_out = _upsample_like(x2_out, x1_out)

                x3_out = self.conv3(x3_up, x2_up)
                x3_out = _upsample_like(x3_out, x1_out)
                                
                x4_out = self.conv4(x4_up, x3_up)
                x4_out = _upsample_like(x4_out, x1_out)

                out = torch.cat((x1_out,x2_out,x3_out,x4_out),1)

                out = self.out(out)

 
        return out
class WATP_DTUM(nn.Module):
    def __init__(self, num_classes):
        super(WATP_DTUM, self).__init__()

        self.WATP = WATP(1,size=512)#这里的32实际上是输出通道数而非num_class
        self.DTUM = DTUM(32, num_classes, num_frames=5)

    def forward(self, X_In, Old_Feat, OldFlag):
      
        FrameNum = X_In.shape[2]##确定帧数
        Features = X_In[:, :, -1, :, :]     #提取图像 BCHW
        Features = self.WATP(Features)      #特征图
        Features = torch.unsqueeze(Features, 2) #重新变成BCTHW
        if OldFlag == 1:  # append current features based on Old Features, for iteration input

            Features = torch.cat([Old_Feat, Features], 2)
        elif OldFlag == 0 and FrameNum > 1:
            for i_fra in range(FrameNum - 1):
                x_t = X_In[:, :, -2 - i_fra, :, :]
                x_t = self.WATP(x_t)
                x_t = torch.unsqueeze(x_t, 2)
                Features = torch.cat([x_t, Features], 2)
                
        Old_Feat=Features[:, :, -(FrameNum -1):, :, :].detach()
        X_Out= self.DTUM(Features)
        return X_Out, Old_Feat

    
class WATP_DTUM_concat(nn.Module):
    def __init__(self, num_classes):
        super(WATP_DTUM_concat, self).__init__()

        self.WATP = WATP(1,size=512)#这里的32实际上是输出通道数而非num_class
        self.DTUM = DTUM_concat(32, num_classes, num_frames=5)

    def forward(self, X_In, Old_Feat, OldFlag):
      
        FrameNum = X_In.shape[2]##确定帧数
        Features = X_In[:, :, -1, :, :]     #提取图像 BCHW
        Features = self.WATP(Features)      #特征图
        Features = torch.unsqueeze(Features, 2) #重新变成BCTHW
        if OldFlag == 1:  # append current features based on Old Features, for iteration input

            Features = torch.cat([Old_Feat, Features], 2)
        elif OldFlag == 0 and FrameNum > 1:
            for i_fra in range(FrameNum - 1):
                x_t = X_In[:, :, -2 - i_fra, :, :]
                x_t = self.WATP(x_t)
                x_t = torch.unsqueeze(x_t, 2)
                Features = torch.cat([x_t, Features], 2)
                
        Old_Feat=Features[:, :, -(FrameNum -1):, :, :].detach()
        X_Out= self.DTUM(Features)
        return X_Out, Old_Feat