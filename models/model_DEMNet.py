import torch
import torch.nn as nn
from models.layers import DTUM,DTUM_concat
import torch.nn.functional as F
from functools import partial
from models.model_WATP import *


def _upsample_like(src, tar):
    #print('str和tar的形状为:'+str(src.shape)+str(tar.shape))
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    x_LL = x1 + x2 + x3 + x4 # actually, this is the average pooling operation with kernel size 2 and stride 2

    return  x_LL

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

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
        self.fuse = fuse(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_mid(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = self.fuse(x1, x2)
        return self.conv(x)
    
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
        self.fuse = fuse(out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)
        if self.bilinear:
            x1 = self.conv_mid(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # x = torch.cat([x2, x1], dim=1)
        x = self.fuse(x1, x2)
        return self.conv(x)
    
class part_dwt(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x_ll = self.dwt(x)
        out = self.conv(x_ll)

        return out


class MIF(nn.Module):
    def __init__(self,DTUM_True,size):
        super(MIF, self).__init__()
        self.n_classes = 1
        self.in_resizer = Resizer()
       # self.saa = SAA(cfg)
        self.layer = 4
        self.mode = 1
        self.n_channels = 3
        self.DTUM_True=DTUM_True
        self.size=size
        if self.mode == 1:
            if self.layer == 4:
                self.inc = DoubleConv(self.n_channels, 32)
                # self.inc = DoubleConv(1, 64)
                self.down1 = (part_dwt(32, 64))
                self.down2 = (part_dwt(64, 128))
                self.down3 = (part_dwt(128, 256))

                self.up4 = (part_iwt1_fuse(64, 256, True))
                # self.conv4 = nn.Conv2d(512, self.n_classes, 1)
                # self.conv4 = (out_conv_last(512, 256, 128, self.n_classes))
                self.conv4 = (out_conv(256, 128, 8))

                self.up3 = (part_iwt1(256, 128, True))
                # self.conv3 = nn.Conv2d(256, self.n_classes, 1)
                self.conv3 = (out_conv(128, 64, 8))
                
                self.up2 = (part_iwt1(128, 64, True))
                # self.conv2 = nn.Conv2d(128, self.n_classes, 1)
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

                x1 = self.down1(x)
           
                x2 = self.down2(x1)

                x3 = self.down3(x2)

                x4_up = self.up4(x1, x3)
                
                x3_up = self.up3(x4_up, x2)
                x2_up = self.up2(x3_up, x1)

                x1_up = self.up1(x2_up, x)

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
    
class DEMNet(nn.Module):
    def __init__(self, num_classes):
        super(DEMNet, self).__init__()

        self.MIF = MIF(1,size=256)#这里的32实际上是输出通道数而非num_class
        self.DTUM = DTUM_concat(16, num_classes, num_frames=5)

    def forward(self, X_In, Old_Feat, OldFlag):
      
        FrameNum = X_In.shape[2]##确定帧数
        Features = X_In[:, :, -1, :, :]     #提取图像 BCHW
        
        Features = self.MIF(Features)      #特征图
        Features = torch.unsqueeze(Features, 2) #重新变成BCTHW
        if OldFlag == 1:  # append current features based on Old Features, for iteration input

            Features = torch.cat([Old_Feat, Features], 2)
            # for i_fra in range(FrameNum-1):
            #     are_equal = torch.allclose(Features[:, :, -1 - i_fra, :, :] , Old_Feat[:, :, -1 - i_fra, :, :] )
            #     print(are_equal) 
            
        elif OldFlag == 0 and FrameNum > 1:
            for i_fra in range(FrameNum - 1):
                x_t = X_In[:, :, -2 - i_fra, :, :]
                x_t = self.MIF(x_t)
                x_t = torch.unsqueeze(x_t, 2)
                Features = torch.cat([x_t, Features], 2)
                
        Old_Feat=Features[:, :, -(FrameNum -1):, :, :].detach()
        # for i_fra in range(FrameNum-1):
        #     are_equal = torch.allclose(Features[:, :, -1 - i_fra, :, :] , Old_Feat[:, :, -1 - i_fra, :, :] )
        #     print(are_equal) 

        X_Out= self.DTUM(Features)
        #print(Old_Feat[:, :, -1, :, :])
        return X_Out, Old_Feat