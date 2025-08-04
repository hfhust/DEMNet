import torch
from torch import nn
import numpy as np
import math
def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace = True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace = True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = out + residual
        return out



class DTUM(nn.Module):    # final version
    def __init__(self, in_channels, num_classes, num_frames):
        super(DTUM, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        # self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), return_indices=True, ceil_mode=False)
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')
        self.relu = nn.ReLU(inplace=True)
        inch = in_channels
        pad = int((num_frames-1)/2)
        self.bn0 = nn.BatchNorm3d(inch)
        self.conv1_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn1_1 = nn.BatchNorm3d(inch)
        self.conv2_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_1 = nn.BatchNorm3d(inch)
        self.conv3_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_1 = nn.BatchNorm3d(inch)
        self.conv4_1 = nn.Conv3d(inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn4_1 = nn.BatchNorm3d(inch)
       
        ##加入层数  代码开始
        self.bn5_1 = nn.BatchNorm3d(inch)
        self.conv4_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn4_2 = nn.BatchNorm3d(inch)
        
        self.bn6_1 = nn.BatchNorm3d(inch)
        self.conv5_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn5_2 = nn.BatchNorm3d(inch)
        ##加入层数  代码结束
        self.conv3_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_2 = nn.BatchNorm3d(inch)
        self.conv2_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_2 = nn.BatchNorm3d(inch)
        self.conv1_2 = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(0,0,0))

        #self.conv1_2_old = nn.Conv3d(2*inch, inch, kernel_size=(num_frames,1,1), padding=(pad,0,0))

        self.bn1_2 = nn.BatchNorm3d(inch)

        self.final = nn.Sequential(
            nn.Conv3d(in_channels=2*inch, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
         )
        self.final_old = nn.Sequential(
            nn.Conv3d(in_channels=2*inch, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
         )

    def direction(self, arr):
        b,c,t,m,n = arr.size()
        #print("m:"+str(m))
        arr[:, :, 1:, :, :] = arr[:, :, 1:, :, :] - m * 2 * n * 2
        arr[:, :, 2:, :, :] = arr[:, :, 2:, :, :] - m * 2 * n * 2
        arr[:, :, 3:, :, :] = arr[:, :, 3:, :, :] - m * 2 * n * 2
        arr[:, :, 4:, :, :] = arr[:, :, 4:, :, :] - m * 2 * n * 2
        #print(arr)
        arr_r_l = arr % 2  # right 1; left 0     [0 1; 0 1]
        up_down = torch.Tensor(range(0,m)).cuda(arr.device) * n*2*2  #.transpose(0,1)
        up_down = up_down.repeat_interleave(n).reshape(m,n)

        arr1 = arr.float() - up_down.reshape([1,1,1,m,n])
        

        arr_u_d = (arr1 >= n*2).float() * 2  # up 0; down 1  [0 0; 2 2]
        arr_out = arr_r_l.float() + arr_u_d   # [0 1; 2 3]1        
        arr_out = (arr_out - 1.5)       # [-1.5 -0.5; 0.5 1.5]
        return arr_out

    def direction1(self, arr):
        b,c,t,m,n = arr.size()
        arr_out = torch.zeros((b, c, t, m, n), device=arr.device)
        #print("m:"+str(m))
        for i in range (0,t-1):
            arr[:, :, (i+1):, :, :] = arr[:, :, (i+1):, :, :] - m * 2 * n * 2
        arr_x=arr%(m*2)
        arr_y=arr//(m*2)

        A=0.5 
        for i in range(t-1):
            mask_left = arr_x[:, :, i, :, :] < arr_x[:, :, -1, :, :]
            mask_right = arr_x[:, :, i, :, :] > arr_x[:, :, -1, :, :]
            l_r = torch.where(mask_left, -1, torch.where(mask_right, 1, 0))

            mask_up = arr_y[:, :, i, :, :] < arr_y[:, :, -1, :, :]
            mask_down = arr_y[:, :, i, :, :] > arr_y[:, :, -1, :, :]
            u_d = torch.where(mask_up, -1, torch.where(mask_down, 1, 0))
            arr_out[:, :, i, :, :] = A*u_d + l_r
        if t > 1:
            arr_last = torch.sum(arr_out[:, :, :-1, :, :], dim=2) / (t - 1)
            arr_out[:, :, -1, :, :] = arr_last
        else:
            arr_out[:, :, -1, :, :] = torch.zeros((b, c, m, n), device=arr.device)        
        return arr_out
    
    def forward(self, x):
        #print("x:"+str(x.shape)) ##([1, 32, 5, 512, 512])
        x = self.relu(self.bn0(x))
        x_1 = self.relu(self.bn1_1(self.conv1_1(x)))##x1:torch.Size([1, 32, 5, 512, 512])
 
        xp_1, ind = self.pool(x_1)
        x_2 = self.relu(self.bn2_1(torch.abs(self.conv2_1(xp_1 * self.direction(ind)))))##x2:torch.Size([1, 32, 5, 256, 256])

        xp_2, ind = self.pool(x_2)
        x_3 = self.relu(self.bn3_1(torch.abs(self.conv3_1(xp_2 * self.direction(ind)))))##x3 :torch.Size([1, 32, 5, 128, 128])

        ##5层DCCB
        # xp_3, ind = self.pool(x_3)
        # x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(xp_3 * self.direction(ind)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        # xp_4, ind = self.pool(x_4)
        # x_5 = self.relu(self.bn5_1(torch.abs(self.conv4_1(xp_4 * self.direction(ind)))))##x4 :torch.Size([1, 32, 5, 64, 64])  
        # xp_5, ind = self.pool(x_5)
        # x_6 = self.relu(self.bn4_1(torch.abs(self.conv4_1(xp_5 * self.direction(ind)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        # o_5 = self.relu(self.bn5_2(self.conv5_2(torch.cat([self.up(x_6),x_5], dim=1))))
        # o_4 = self.relu(self.bn4_2(self.conv4_2(torch.cat([self.up(o_5),x_4], dim=1))))
        # o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(o_4),x_3], dim=1)))) ##从4层到5层需要替换这一行
        # o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        # o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))  ##o1:torch.Size([1, 32, 1, 512, 512])
        ##4层DCCB
        # xp_3, ind = self.pool(x_3)
        # x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(xp_3 * self.direction(ind)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        # xp_4, ind = self.pool(x_4)
        # x_5 = self.relu(self.bn5_1(torch.abs(self.conv4_1(xp_4 * self.direction(ind)))))##x4 :torch.Size([1, 32, 5, 64, 64]) 
        # o_4 = self.relu(self.bn4_2(self.conv4_2(torch.cat([self.up(x_5),x_4], dim=1))))
        # o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(o_4),x_3], dim=1)))) ##从4层到5层需要替换这一行
        # o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        # o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))  ##o1:torch.Size([1, 32, 1, 512, 512])
        # ##3层DCCB
        xp_3, ind = self.pool(x_3)
        x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(xp_3 * self.direction(ind)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(x_4),x_3], dim=1)))) ##从4层到5层需要替换这一行
        o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1)))) ##o1:torch.Size([1, 32, 1, 512, 512]) 
        ##2层DCCB
        # o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(x_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        # o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))  ##o1:torch.Size([1, 32, 1, 512, 512])
        # x_old = self.relu(self.bn1_2(self.conv1_2_old(torch.cat([self.up(o_2),x_1], dim=1))))
        # x_old = x[:,:,:4,:,:]
        # x_old = x_old.detach()
        x_out = self.final(torch.cat([o_1, torch.unsqueeze(x[:,:,-1,:,:],2)], dim=1))##xout:torch.Size([1, 1, 1, 512, 512])


        return x_out

class DTUM_concat(nn.Module):    # final version
    def __init__(self, in_channels, num_classes, num_frames):
        super(DTUM_concat, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2), padding=(0,0,0), return_indices=True)
        self.up = nn.Upsample(scale_factor=(1,2,2), mode='nearest')
        inch =[16,32,64,128,256,512]
        self.relu = nn.ReLU(inplace=True)
        pad = int((num_frames-1)/2)
        self.bn0 = nn.BatchNorm3d(inch[0])
        self.conv1_1 = nn.Conv3d(inch[0], inch[0], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn1_1 = nn.BatchNorm3d(inch[0])
        self.conv2_1 = nn.Conv3d(2*inch[0], inch[1], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_1 = nn.BatchNorm3d(inch[1])
        self.conv3_1 = nn.Conv3d(2*inch[1], inch[2], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_1 = nn.BatchNorm3d(inch[2])
        self.conv4_1 = nn.Conv3d(2*inch[2], inch[3], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn4_1 = nn.BatchNorm3d(inch[3])
       
        self.conv5_1 = nn.Conv3d(2*inch[3], inch[4], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn5_1 = nn.BatchNorm3d(inch[4])
        self.conv6_1 = nn.Conv3d(2*inch[4], inch[5], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn6_1 = nn.BatchNorm3d(inch[5])

        self.conv5_2 = nn.Conv3d(inch[5]+inch[4], inch[4], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn5_2 = nn.BatchNorm3d(inch[4])        
        self.conv4_2 = nn.Conv3d(inch[4]+inch[3], inch[3], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn4_2 = nn.BatchNorm3d(inch[3])

        self.conv3_2 = nn.Conv3d(inch[3]+inch[2], inch[2], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn3_2 = nn.BatchNorm3d(inch[2])
        self.conv2_2 = nn.Conv3d(inch[2]+inch[1], inch[1], kernel_size=(num_frames,1,1), padding=(pad,0,0))
        self.bn2_2 = nn.BatchNorm3d(inch[1])
        self.conv1_2 = nn.Conv3d(inch[1]+inch[0], inch[0], kernel_size=(num_frames,1,1), padding=(0,0,0))

        self.bn1_2 = nn.BatchNorm3d(inch[0])

        self.final = nn.Sequential(
            nn.Conv3d(in_channels=2*inch[0], out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Dropout3d(0.5),
            nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
         )
        # self.final_old = nn.Sequential(
        #     nn.Conv3d(in_channels=2*inch, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        #     nn.BatchNorm3d(32), nn.ReLU(),
        #     nn.Dropout3d(0.5),
        #     nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        #  )
        #nn.Conv3d(in_channels=2*inch, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        #     nn.BatchNorm3d(32), nn.ReLU(),
        #     nn.Dropout3d(0.5),
        #     nn.Conv3d(in_channels=32, out_channels=num_classes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
    def direction1(self, arr):
            b,c,t,m,n = arr.size()
            #print("m:"+str(m))
            arr[:, :, 1:, :, :] = arr[:, :, 1:, :, :] - m * 2 * n * 2
            arr[:, :, 2:, :, :] = arr[:, :, 2:, :, :] - m * 2 * n * 2
            arr[:, :, 3:, :, :] = arr[:, :, 3:, :, :] - m * 2 * n * 2
            arr[:, :, 4:, :, :] = arr[:, :, 4:, :, :] - m * 2 * n * 2
            #print(arr)
            arr_r_l = arr % 2  # right 1; left 0     [0 1; 0 1]
            arr_out = torch.zeros((b, c, t, m, n), device=arr.device)
            up_down = torch.arange(0, m, device=arr.device) * n * 2 * 2 #.transpose(0,1)
            up_down = up_down.repeat_interleave(n).reshape(m,n)

            arr1 = arr.float() - up_down.reshape([1,1,1,m,n])
            

            arr_u_d = (arr1 >= n*2).float() * 2  # up 0; down 1  [0 0; 2 2]
            arr_out = arr_r_l.float() + arr_u_d   # [0 1; 2 3]1        
            # arr_out = (arr_out - 1.5) 
            arr_out = (arr_out/4 +1.25)     
            return arr_out
    def direction2(self, arr):
        b,c,t,m,n = arr.size()
        arr_out = torch.zeros((b, c, t, m, n), device=arr.device)
        #print("m:"+str(m))
        for i in range (0,t-1):
            arr[:, :, (i+1):, :, :] = arr[:, :, (i+1):, :, :] - m * 2 * n * 2
        arr_x=arr%(m*2)
        arr_y=arr//(m*2)

        A=0.8
        for i in range(t-1):
            mask_left = arr_x[:, :, i, :, :] < arr_x[:, :, -1, :, :]
            mask_right = arr_x[:, :, i, :, :] > arr_x[:, :, -1, :, :]
            l_r = torch.where(mask_left, -1, torch.where(mask_right, 1, 0))

            mask_up = arr_y[:, :, i, :, :] < arr_y[:, :, -1, :, :]
            mask_down = arr_y[:, :, i, :, :] > arr_y[:, :, -1, :, :]
            u_d = torch.where(mask_up, -1, torch.where(mask_down, 1, 0))
            
            intermediate_value=A*u_d + l_r
            # mask_positive = intermediate_value >= 0
            # mask_negative = intermediate_value < 0 
            
            # # 对非负部分应用sqrt
            # positive_part = torch.sqrt(intermediate_value)
            
            # # 对负数部分，先取绝对值再开方，并加负号
            # negative_part = -torch.sqrt(torch.abs(intermediate_value))
            
            # 使用torch.where来合并正负部分
            arr_out[:, :, i, :, :] = torch.where(
                    intermediate_value >= 0,  # 条件：是否为非负
                     torch.sqrt(intermediate_value),  # 条件成立时的值：直接开平方根
                    -torch.sqrt(torch.abs(intermediate_value))  # 条件不成立时的值：取绝对值开平方根后加负号
                                                )
            # arr_out[:, :, i, :, :] = A*u_d + l_r
        if t > 1:
            arr_last = torch.sum(arr_out[:, :, :-1, :, :], dim=2) / (t - 1)
            arr_out[:, :, -1, :, :] = arr_last
        else:
            arr_out[:, :, -1, :, :] = torch.zeros((b, c, m, n), device=arr.device)        
        return arr_out
    def forward(self, x):
        #print("x:"+str(x.shape)) ##([1, 32, 5, 512, 512])
        x = self.relu(self.bn0(x))
        x_1 = self.relu(self.bn1_1(self.conv1_1(x)))##x1:torch.Size([1, 32, 5, 512, 512])
 
        xp_1, ind = self.pool(x_1)
        x_2 = self.relu(self.bn2_1(torch.abs(self.conv2_1(torch.cat((xp_1* self.direction1(ind),self.direction2(ind)), dim=1)))))##x2:torch.Size([1, 32, 5, 256, 256])

        xp_2, ind = self.pool(x_2)
        x_3 = self.relu(self.bn3_1(torch.abs(self.conv3_1(torch.cat((xp_2 * self.direction1(ind),self.direction2(ind)), dim=1)))))##x3 :torch.Size([1, 32, 5, 128, 128])

        ##5层DCCB
        # xp_3, ind = self.pool(x_3)
        # x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(torch.cat((xp_3* self.direction1(ind),self.direction2(ind)), dim=1)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        # xp_4, ind = self.pool(x_4)
        # x_5 = self.relu(self.bn5_1(torch.abs(self.conv5_1(torch.cat((xp_4* self.direction1(ind),self.direction2(ind)), dim=1)))))##x4 :torch.Size([1, 32, 5, 64, 64])  
        # xp_5, ind = self.pool(x_5)
        # x_6 = self.relu(self.bn6_1(torch.abs(self.conv6_1(torch.cat((xp_5* self.direction1(ind),self.direction2(ind)), dim=1)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        # o_5 = self.relu(self.bn5_2(self.conv5_2(torch.cat([self.up(x_6),x_5], dim=1))))
        # o_4 = self.relu(self.bn4_2(self.conv4_2(torch.cat([self.up(o_5),x_4], dim=1))))
        # o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(o_4),x_3], dim=1)))) ##从4层到5层需要替换这一行
        # o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        # o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))  ##o1:torch.Size([1, 32, 1, 512, 512])
        ##4层DCCB
        xp_3, ind = self.pool(x_3)
        x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(torch.cat((xp_3* self.direction1(ind),self.direction2(ind)), dim=1)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        xp_4, ind = self.pool(x_4)
        x_5 = self.relu(self.bn5_1(torch.abs(self.conv5_1(torch.cat((xp_4* self.direction1(ind),self.direction2(ind)), dim=1)))))##x4 :torch.Size([1, 32, 5, 64, 64]) 
        o_4 = self.relu(self.bn4_2(self.conv4_2(torch.cat([self.up(x_5),x_4], dim=1))))
        o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(o_4),x_3], dim=1)))) ##从4层到5层需要替换这一行
        o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))  ##o1:torch.Size([1, 32, 1, 512, 512])
        # ##3层DCCB
        # xp_3, ind = self.pool(x_3)
        # x_4 = self.relu(self.bn4_1(torch.abs(self.conv4_1(torch.cat((xp_3* self.direction1(ind),self.direction2(ind)), dim=1)))))##x4 :torch.Size([1, 32, 5, 64, 64])
        # o_3 = self.relu(self.bn3_2(self.conv3_2(torch.cat([self.up(x_4),x_3], dim=1)))) ##从4层到5层需要替换这一行
        # o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(o_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        # o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1)))) ##o1:torch.Size([1, 32, 1, 512, 512]) 
        ##2层DCCB
        # o_2 = self.relu(self.bn2_2(self.conv2_2(torch.cat([self.up(x_3),x_2], dim=1)))).detach() ##o2:torch.Size([1, 32, 5, 256, 256])
        # o_1 = self.relu(self.bn1_2(self.conv1_2(torch.cat([self.up(o_2),x_1], dim=1))))  ##o1:torch.Size([1, 32, 1, 512, 512])
        # x_old = self.relu(self.bn1_2(self.conv1_2_old(torch.cat([self.up(o_2),x_1], dim=1))))
        # x_old = x[:,:,:4,:,:]
        # x_old = x_old.detach()
        x_out = self.final(torch.cat([o_1, torch.unsqueeze(x[:,:,-1,:,:],2)], dim=1))##xout:torch.Size([1, 1, 1, 512, 512])


        return x_out    