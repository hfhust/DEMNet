import os
import torch
import torch.nn as nn
import numpy as np
import math
from skimage import measure
import matplotlib.pyplot as plt


class ShootingRules_A(nn.Module):
    def __init__(self):
        super(ShootingRules_A, self).__init__()

        return
    def forward(self, output, target, DetectTh):      # , mixdata
        target_np = target.data.cpu().numpy().copy()
        output_np = output.data.cpu().numpy().copy()
        # mixdata_np = mixdata.data.cpu().numpy()

        FalseNum=0 #False number
        TrueNum=0 #True number
        TgtNum = 0
        Fat=0
        # DetectTh=0.5 #The detecting threshold used in output
        LocLen1 = 1
        LocLen2 = 4

        for i_batch in range(output_np.shape[0]):
            output_one = output_np[i_batch,-1,:,:]
            target_one = target_np[i_batch,0,:,:]
            # mixdata_one = mixdata_np[i_batch, 0, :, :]

            '''
            fig=plt.figure()
            plt.subplot(221); plt.imshow(np.squeeze(mixdata_one), cmap='gray')
            plt.subplot(222); plt.imshow(np.squeeze(target_one), cmap='gray')
            plt.subplot(223); plt.imshow(np.squeeze(output_one), cmap='gray')
            plt.show()
            '''

            output_one[np.where(output_one < DetectTh)] = 0
            output_one[np.where(output_one >= DetectTh)] = 1
            # output_two=output_one

            # predimage = measure.label(output_two, connectivity=2)  # 标记8连通区域

            # fa_props = measure.regionprops(predimage, intensity_image=output_two, cache=True)

            


            # for i_pred in range(len(fa_props)):
            #     False_flag = 1

            #     pixel_coords = fa_props[i_pred].coords
            #     for i_pixel in pixel_coords:
            #         pred_area = target_one[i_pixel[0]-LocLen1:i_pixel[0]+LocLen1+1, i_pixel[1]-LocLen1:i_pixel[1]+LocLen1+1]
            #         if pred_area.sum() >= 1:
            #             False_flag = 0
            #     if False_flag == 1:
            #         Fat += 1




            labelimage = measure.label(target_one, connectivity=2)  # 标记8连通区域
            props = measure.regionprops(labelimage, intensity_image=target_one, cache=True)     #测量标记连通区域的属性

            # TgtNum += len(props)
            #####################################################################
            # according to label(the lightest pixels)

            Box2_map = np.ones(output_one.shape)
            for i_tgt in range(len(props)):
                # True_flag = 0

                pixel_coords = props[i_tgt].coords
                for i_pixel in pixel_coords:
                    Box2_map[i_pixel[0]-LocLen2:i_pixel[0]+LocLen2+1, i_pixel[1]-LocLen2:i_pixel[1]+LocLen2+1] = 0
                    # Tgt_area = output_one[i_pixel[0]-LocLen1:i_pixel[0]+LocLen1+1, i_pixel[1]-LocLen1:i_pixel[1]+LocLen1+1]
                    # if Tgt_area.sum() >= 1:
                    #     True_flag = 1
                #         break
                # if True_flag == 1:
                #     TrueNum += 1
                #     break         
            False_output_one = output_one*Box2_map
            FalseNum += np.count_nonzero(False_output_one)


        return  FalseNum
    
class ShootingRules_B(nn.Module):
    def __init__(self):
        super(ShootingRules_B, self).__init__()

        return
    def forward(self, output, target, DetectTh):      # , mixdata
        target_np = target.data.cpu().numpy().copy()
        output_np = output.data.cpu().numpy().copy()
        # mixdata_np = mixdata.data.cpu().numpy()
        FalseNum=0 #False number
        TrueNum=0 #True number
        TgtNum = 0
        Fat=0
        # DetectTh=0.5 #The detecting threshold used in output


        for i_batch in range(output_np.shape[0]):
            output_one = output_np[i_batch,-1,:,:]
            target_one = target_np[i_batch,0,:,:]
            # mixdata_one = mixdata_np[i_batch, 0, :, :]
            '''
            fig=plt.figure()
            plt.subplot(221); plt.imshow(np.squeeze(mixdata_one), cmap='gray')
            plt.subplot(222); plt.imshow(np.squeeze(target_one), cmap='gray')
            plt.subplot(223); plt.imshow(np.squeeze(output_one), cmap='gray')
            plt.show()
            '''

            output_one[np.where(output_one < DetectTh)] = 0
            output_one[np.where(output_one >= DetectTh)] = 1

            # 对目标图像和输出图像分别进行连通区域标记
            target_label = measure.label(target_one, connectivity=2)
            pred_label = measure.label(output_one, connectivity=2)

            # 获取目标和预测区域的属性
            target_props = measure.regionprops(target_label, cache=True)
            pred_props = measure.regionprops(pred_label, cache=True)
            TgtNum = len(target_props)           

            matched_targets = set()
            for pred in pred_props:
                pred_centroid = np.array(pred.centroid)
                min_distance = float('inf')
                closest_target = None

            # 找到最近的目标连通区域
                for target in target_props:
                    if target.label not in matched_targets:
                        target_centroid = np.array(target.centroid)
                        distance = np.linalg.norm(pred_centroid - target_centroid)

                        if distance < min_distance:
                            min_distance = distance
                            closest_target = target

                # 如果最近的目标连通区域的距离小于阈值，则认为检测到了该目标
                if closest_target is not None and min_distance <= 3:
                    TrueNum += 1
                    matched_targets.add(closest_target.label)
                if  closest_target is None or min_distance>3:
                    # closest_target is None and min_distance>3 and pred.area>2:
                    Fat += 1
                    FalseNum+=pred.area
                


                      # 标记为已匹配

        return   TrueNum, TgtNum,Fat,FalseNum




