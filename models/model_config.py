import torch
from thop import profile   
from thop import clever_format  
import time 

from models.model_DEMNet import DEMNet
import torch.nn as nn



def model_chose(model, loss_func, SpatialDeepSup):
    num_classes = 1
    
    if model =='DEMNet':  
        net = DEMNet(num_classes=num_classes)   
    
    else:
        raise ValueError(f"Unsupported model: {model}")
    return net



def run_model(mode,net, model, SeqData, Old_Feat, OldFlag):

    # Old_Feat = SeqData[:,:,:-1, :,:] * 0  # interface for iteration input
    # OldFlag = 1  # 1: i

    if model=='DNANet' or model=='ResUNet' or model=='ACM' or model=='ALCNet' or model=='Mamba' or model=='MiM' or model=='WATP' or model=='MSH'   :   
        input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        outputs = net(input)
    elif model =='SC':
        input = SeqData[:, :, -1, :, :]
        outputs = net(input)
    elif model=='DNANet_DTUM' or model=='ResUNet_DTUM' or model=='ALCNet_DTUM' or model=='MiM_DTUM' or model=='ResUNet_DTUM_1' or model=='WATP_DTUM'  or model=='WATP_DTUM_1' :
        input = SeqData.repeat(1, 3, 1, 1, 1)
        outputs,new_Old_Feat = net(input, Old_Feat, OldFlag)

    elif model=='DEMNet' :
        input = SeqData.repeat(1, 3, 1, 1, 1)
        outputs,new_Old_Feat = net(input, Old_Feat, OldFlag)

    elif model == 'UIU':
        input = SeqData[:, :, -1, :, :].repeat(1, 3, 1, 1)
        d0, d1,d2,d3,d4,d5,d6 = net(input)
        outputs = [d0, d1, d2, d3, d4, d5, d6]
    elif model == 'UIU_DTUM':
        input = SeqData.repeat(1, 3, 1, 1, 1)
        d0, d1,d2,d3,d4,d5,d6 = net(input, Old_Feat, OldFlag)
        outputs = [d0, d1, d2, d3, d4, d5, d6]

    # if OldFlag == 0:
    #     if 'DTUM' in model:
    #         if model=='DTUMs':
    #             flops, params = profile(net, inputs=(input, ))
    #         else:
    #             device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")

    #             file_input=torch.zeros([1, 16, 4, 256, 256])
    #             Old_Feat=torch.zeros([1, 16, 4, 256, 256])
    #             flops, params = profile(net, inputs=(input, Old_Feat, 1))   # runtimeerror cpu : net.module
    #     flops, params = clever_format([flops, params], '%.5f')
    #     print(model,'的计算量为：',flops, params)
    if 'DTUM'in model:
        return outputs,new_Old_Feat
    else :
        return outputs
