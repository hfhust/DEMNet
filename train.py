import argparse
import torch
import torch.nn as nn
import ast
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR,CosineAnnealingLR
from numpy import *
import os
import numpy as np
import scipy.io as scio
import time
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import auc
import re
from MIRSDTDataLoader import TrainSetLoader, TestSetLoader
from DAUBDataLoader import DAUB_TestSetLoader, DAUB_TrainSetLoader

from models.model_config import model_chose, run_model
from losses.loss_config import loss_chose
from ShootingRules import ShootingRules_A,ShootingRules_B
from write_results import writeNUDTMIRSDT_ROC, writeIRSeq_ROC
from losses.loss_fullySupervised import rou2



# def str2bool(v):
#     return v.lower() in ('yes', 'true', 't', '1')


def generate_savepath(args, epoch, epoch_loss):

    timestamp = time.time()
    CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime(timestamp))

    SavePath = args.saveDir + args.model +'_'+args.dataset+'_'+args.loss_func+'_'+'align_'+str(args.align)+'/'
    ModelPath = SavePath + 'net_' + str(epoch+1) + '_epoch_' + str(epoch_loss) + '_loss_' + CurTime + '.pth'
    ParameterPath = SavePath + 'net_para_' + CurTime + '.pth'

    if not os.path.exists(args.saveDir):
        os.mkdir(args.saveDir)
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)

    return ModelPath, ParameterPath, SavePath


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Infrared_target_detection_overall')
    parser.add_argument('--DataPath',  type=str, default='./dataset/', help='Dataset path [default: ./dataset/]')
    parser.add_argument('--dataset',   type=str, default='DAUB_DTUM', help='Dataset name [DAUB_DTUM or NUDT-MIRSDT]')
    parser.add_argument('--training_rate', type=int, default=1, help='Rate of samples in training (1/n) [default: 1]')
    parser.add_argument('--saveDir',   type=str, default='./results_FAT/',
                            help='Save path [defaule: ./results/]')
    parser.add_argument('--pth_path', type=str, default='NUDT_best.pth', help='Trained model path')
    parser.add_argument('--train',    type=int, default=0)
    parser.add_argument('--test',     type=int, default=1)
    parser.add_argument('--align', type=ast.literal_eval, default=True, help='Whether to align input frames (True/False)')
    # train
    parser.add_argument('--model',     type=str, default='ResUNet_DTUM',
                        help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU')
    parser.add_argument('--loss_func', type=str, default='fullySup',
                        help='HPM, FocalLoss, OHEM, fullySup, fullySup1(ISNet), fullySup2(UIU),Dice')
    parser.add_argument('--opt', type=str, default='Adam',
                        help='Adam,AdaGrad')
    parser.add_argument('--fullySupervised', default=True)
    parser.add_argument('--SpatialDeepSup',  default=False)
    parser.add_argument('--batchsize', type=int,   default=2)
    parser.add_argument('--epochs',    type=int,   default=20)
    parser.add_argument('--lrate',     type=float, default=0.001)
    parser.add_argument("--lrate_min", type=float, default=1e-5, help="")
    parser.add_argument("--scheduler_settings", default={'epochs': 20, 'min_lr': 1E-5}, type=dict, help="scheduler settings")
    parser.add_argument('--pretrained_model_path',type=str, default=None)
    # loss
    parser.add_argument('--MyWgt',     default=[0.1667, 0.8333], help='Weights of positive and negative samples')
    parser.add_argument('--MaxClutterNum', type=int, default=39, help='Clutter samples in loss [default: 39]')
    parser.add_argument('--ProtectedArea', type=int, default=2,  help='1,2,3...')

    # GPU
    parser.add_argument('--DataParallel',     default=False,    help='Use one gpu or more')
    parser.add_argument('--device', type=str, default="cuda:0", help='use comma for multiple gpus')

    args = parser.parse_args()
    # the parser
    return args




class Trainer(object):
    def __init__(self, args):
        self.args = args
        # model
        self.net = model_chose(args.model, args.loss_func, args.SpatialDeepSup)

        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)



        train_path = args.DataPath + args.dataset + '/'
        self.test_path = train_path
        if args.dataset == 'NUDT-MIRSDT':
            self.train_dataset = TrainSetLoader(train_path, fullSupervision=args.fullySupervised)
            self.val_dataset = TestSetLoader(self.test_path)
        elif args.dataset == 'DAUB_DTUM':
            self.train_dataset = DAUB_TrainSetLoader(train_path, fullSupervision=args.fullySupervised, align=args.align)
            self.val_dataset = DAUB_TestSetLoader(self.test_path, align=args.align)

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batchsize, shuffle=True, drop_last=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, )
        if args.opt=='Adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=args.lrate, betas=(0.9, 0.99))
            # self.scheduler = StepLR(self.optimizer, step_size=3, gamma=0.5, last_epoch=-1)#学习率衰减
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.scheduler_settings['min_lr'])

        elif args.opt=='AdaGrad':
            self.optimizer = optim.Adagrad(self.net.parameters(), lr=0.06,weight_decay=4e-4)
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=args.scheduler_settings['min_lr'])

        self.criterion = loss_chose(args)
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.criterion1=nn.BCEWithLogitsLoss(size_average=True)
        self.eval_metrics = ShootingRules_B()
        self.AUC = ShootingRules_A()
        self.MSE = nn.MSELoss()

        self.loss_list = []
        self.Gain = 100
        self.epoch_loss = 0

        ########### save ############
        self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(args, 0, 0)
        self.test_save = self.SavePath[0:-1] + '_visualization/'
        self.writeflag = 1
        self.save_flag = 1
        if self.save_flag == 1 and not os.path.exists(self.test_save):
            os.mkdir(self.test_save)
        
        self.seq_numbers = None
        if hasattr(args, 'pretrained_model_path') and args.pretrained_model_path:
            self.load_pretrained_weights(args.pretrained_model_path)

    def get_sequence_number(self, path, dataset):
        if 'IRDST_DTUM' in dataset:
            return int(path.split('/')[5])
        elif 'DAUB_DTUM' in dataset:
            return int(path.split('data')[-1].split('/images')[0])
        elif 'zqr' in dataset:
            return int(path.split('data')[-1].split('/images')[0])
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    def load_sequence_numbers(self, file_path, dataset):
        txt = np.loadtxt(file_path, dtype=bytes).astype(str)
        return [self.get_sequence_number(path, dataset) for path in txt]
    
    def load_sequence_numberslist(self, file_path, dataset):
        txt = np.loadtxt(file_path, dtype=bytes).astype(str)
        seen = set()
        unique_seq_numbers = []
        for path in txt:
            seq_num = self.get_sequence_number(path, dataset)
            if seq_num not in seen:
                seen.add(seq_num)
                unique_seq_numbers.append(seq_num)
        return unique_seq_numbers

    def load_pretrained_weights(self, model_path):
        """
        加载预训练模型权重
        """
        model_dict = self.net.state_dict()
        pretrained_dict = torch.load(model_path, map_location=self.device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and model_dict[k].shape == v.shape:
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        self.net.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\n")

    def training(self, epoch):
        args = self.args
        running_loss = 0.0
        loss_last = 0.0
        self.net.train()
        current_seq_num=None
        OldFlag=0
        if 'IRDST_DTUM' in args.dataset:
            self.seq_numbers = self.load_sequence_numbers('/home/yons/data1/IRDST_DTUM/train_IRDST.txt', args.dataset)
        elif 'DAUB_DTUM' in args.dataset:
            self.seq_numbers = self.load_sequence_numbers('/home/yons/data1/DAUB_DTUM/train_DAUB.txt', args.dataset)
        for i, data in enumerate(tqdm(self.train_loader), 0):
            # if i % args.training_rate != 0:  ## 数据量很大时可以跳过一些数据
            #     continue
            # if i>10:break  
            if 'NUDT-MIRSDT' in args.dataset:
                if i % 100 == 0:
                    OldFlag = 0
            else:
                Seq_num = self.seq_numbers[i]
                if current_seq_num is None or Seq_num != current_seq_num:  # 序列编号改变，重置计数器
                    current_seq_num = Seq_num
                    OldFlag = 0
            SeqData_t, TgtData_t, m, n = data
            SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)# b,t,m,n  // b,1,m.n
            if OldFlag == 0:
                Old_Feat = torch.zeros([1, 32, 4, 512, 512]).to(self.device)
            else:
                Old_Feat = Old_Feat

                # 保持上一次的值
            if 'DTUM' in args.model:
                outputs,Old_Feat = run_model(1,self.net, args.model, SeqData,Old_Feat,OldFlag)
                OldFlag=0
            else:
                outputs = run_model(1,self.net, args.model, SeqData,Old_Feat,OldFlag)
            if isinstance(outputs, list):
                if isinstance(outputs[0], tuple):
                    outputs[0] = outputs[0][0]
            elif isinstance(outputs, tuple):
                outputs = outputs[0]

            if 'DNANet' in args.model:
                loss = 0
                if isinstance(outputs, list):
                    for output in outputs:
                        loss += self.criterion(output, TgtData.float())
                    loss /= len(outputs)
                else:
                    loss = self.criterion(outputs, TgtData.float())
            elif 'UIU' in args.model:
                if 'fullySup2' in args.loss_func:
                    loss0, loss = self.criterion(outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], TgtData.float())
                    if not args.SpatialDeepSup:
                        loss = loss0   ## without SDS
                else:
                    loss = 0
                    if not args.SpatialDeepSup:
                        loss = self.criterion(outputs[0], TgtData.float())
                    else:
                        for output in outputs:
                            loss += self.criterion(output, TgtData.float())
            else :
                loss = self.criterion(outputs, TgtData.float())

            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()


        self.epoch_loss = running_loss / i * self.Gain
        print('model: %s, epoch: %d, loss: %.10f' % (args.model + args.loss_func, epoch + 1, self.epoch_loss))
        ########################################
        self.scheduler.step()
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < args.lrate_min:
            self.optimizer.param_groups[0]['lr'] = args.lrate_min
        self.loss_list.append(self.epoch_loss)


    def validation(self, epoch):
        args = self.args                  
        if 'NUDT-MIRSDT' in args.dataset:
            txt = np.loadtxt(self.test_path + 'test.txt', dtype=bytes).astype(str)
            self.net.eval()
            low_snr3 = [47,56,59,76,92,101,105,119]
            high_snr3 = [85,86,87,88,89,90,91,93,94,95,96,97]


            Th_Seg = np.array([0.5])##置信阈值
            Th_Seg = np.array(
            [0, 1e-30, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
             1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, .15, 0.2, .25, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, .75,
             0.8, .85, 0.9, 0.95, 0.975, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999, 0.999999, 0.9999999, 1])
            if epoch < args.epochs-1:
                Th_Seg = np.array([0, 1e-1, 0.2, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.9, 0.95, 1])##训练时置信阈值设置
            FalseNumAll = np.zeros([20,len(Th_Seg)])
            TrueNumAll = np.zeros([20,len(Th_Seg)])
            TgtNumAll = np.zeros([20,len(Th_Seg)])
            FatNumAll=np.zeros([20,len(Th_Seg)])
            
            OldFlag = 0
            Old_Feat = torch.zeros([1,32,4,512,512]).to(self.device)  # interface for iteration input
            pixelsNumber = np.zeros(20)
            time_start = time.time()
            for i, data in enumerate(tqdm(self.val_loader), 0):
                if i % 100 == 0:
                    OldFlag = 0
                else:
                    OldFlag = 1
                Seq_num = int(txt[i].split('Sequence')[1].split('/Mix')[0])
                index = (low_snr3+high_snr3).index(Seq_num)

                with torch.no_grad():
                    SeqData_t, TgtData_t, m, n = data
                    SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)
                    if 'DTUM' in args.model:
                        outputs,Old_Feat = run_model(0,self.net, args.model, SeqData,Old_Feat,OldFlag)
                        OldFlag=1
                    else:
                        outputs = run_model(0,self.net, args.model, SeqData,Old_Feat,OldFlag)

                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    outputs = torch.squeeze(outputs, 2)
                    Outputs_Max = torch.sigmoid(outputs)
                    TestOut = Outputs_Max.data.cpu().numpy()[0, 0, 0:m, 0:n]

                    pixelsNumber[index] += m * n
                    if self.save_flag:
                        img = Image.fromarray(uint8(TestOut * 255))
                        folder_name = "%sSequence%d/" % (self.test_save, Seq_num)
                        if not os.path.exists(folder_name):
                            os.mkdir(folder_name)
                        name = folder_name + ('%05d.png' % (i % 100 + 1))
                        img.save(name)
                    # the statistics for detection result
                    if self.writeflag:
                        for th_i in range(len(Th_Seg)):
                            TrueNum, TgtNum,Fat,FalseNum = self.eval_metrics(Outputs_Max[:,:,:m,:n], TgtData[:,:,:m,:n], Th_Seg[th_i])
                            FalseNumAll[index, th_i] = FalseNumAll[index, th_i] + FalseNum       #源码
                            TrueNumAll[index, th_i] = TrueNumAll[index, th_i] + TrueNum          #源码
                            TgtNumAll[index, th_i] = TgtNumAll[index, th_i] + TgtNum           #源码
                            FatNumAll[index, th_i] = FatNumAll[index, th_i] + Fat

            time_end = time.time()
            print('FPS=%.3f' % ((i+1)/(time_end-time_start)))

            if self.writeflag:

                Pd_lSNR = np.sum(TrueNumAll[0:8, :], axis=0) / np.sum(TgtNumAll[0:8, :], axis=0)
                Pd_hSNR = np.sum(TrueNumAll[8:, :], axis=0) / np.sum(TgtNumAll[8:, :], axis=0)
                Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
                Fa_lSNR = np.sum(FalseNumAll[0:8, :], axis=0) / pixelsNumber[0:8].sum()
                Fa_hSNR = np.sum(FalseNumAll[8:, :], axis=0) / pixelsNumber[8:].sum()
                Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()

                Fat_all= np.sum(FatNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)

                writelines = open(self.SavePath + 'Epoch' + str(epoch+1) + '_ROC_ShootingRules.txt', 'w')
                for i in range(20):
                    seq = (low_snr3+high_snr3)[i]
                    writelines.write('Seq' + str(seq) + 'results:\n')
                    for seg_i in range(len(Th_Seg)):
                        writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[seg_i], TrueNumAll[i, seg_i], TgtNumAll[i, seg_i],
                                    TrueNumAll[i, seg_i] / TgtNumAll[i, seg_i], FalseNumAll[i, seg_i], FalseNumAll[i, seg_i] / pixelsNumber[i],
                                    FatNumAll[i, seg_i],TgtNumAll[i, seg_i],FatNumAll[i, seg_i] / TgtNumAll[i, seg_i]))
                writelines.write('Low SNR results:\n')
                for th_i in range(len(Th_Seg)):
                    writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[th_i], 
                                    TrueNumAll[0:8:, th_i].sum(),TgtNumAll[0:8:, th_i].sum(),TrueNumAll[0:8:, th_i].sum() / TgtNumAll[0:8:, th_i].sum(),
                                    FalseNumAll[0:8:, th_i].sum(), FalseNumAll[0:8:, th_i].sum() / pixelsNumber[0:8].sum(),
                                    FatNumAll[0:8:, th_i].sum(),TgtNumAll[0:8:, th_i].sum(),FatNumAll[0:8:, th_i].sum() / TgtNumAll[0:8:, th_i].sum(),
                                    ))

                writelines.write('High SNR results:\n')
                for th_i in range(len(Th_Seg)):
                    writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[th_i], 
                                    TrueNumAll[8:, th_i].sum(),TgtNumAll[8:, th_i].sum(),TrueNumAll[8:, th_i].sum() / TgtNumAll[8:, th_i].sum(),
                                    FalseNumAll[8:, th_i].sum(), FalseNumAll[8:, th_i].sum() / pixelsNumber[8:].sum(),
                                    FatNumAll[8:, th_i].sum(),TgtNumAll[8:, th_i].sum(),FatNumAll[8:, th_i].sum() / TgtNumAll[8:, th_i].sum(),
                                    ))

                writelines.write('Final results:\n')
                for th_i in range(len(Th_Seg)):
                    writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[th_i], 
                                    TrueNumAll[:, th_i].sum(),TgtNumAll[:, th_i].sum(),TrueNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                                    FalseNumAll[:, th_i].sum(), FalseNumAll[:, th_i].sum() / pixelsNumber.sum(),
                                    FatNumAll[:, th_i].sum(),TgtNumAll[:, th_i].sum(),FatNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                                    ))
                writelines.close()
                seg = 0
                seg = 29 #seg根据置信度阈值设置而更换
                if epoch < args.epochs-1:
                    seg = 7
                
                print('model: %s, epoch: %d, Th_Seg = %.4e, PD:[%d, %.5f], FA:[%d, %.4e],FAT:[%d, %.5f]' % (args.model + args.loss_func, epoch + 1,
                    Th_Seg[seg], TrueNumAll[:,seg].sum(), Pd_all[seg], FalseNumAll[:,seg].sum(), Fa_all[seg],FatNumAll[:,seg].sum(),Fat_all[seg]))
        elif 'DAUB_DTUM' in args.dataset: 
            self.seq_numbers = self.load_sequence_numbers('./dataset/DAUB_DTUM/test_DAUB.txt', args.dataset)

            self.seq_numbers_list = self.load_sequence_numberslist('./dataset/DAUB_DTUM/test_DAUB.txt', args.dataset)
            seq_to_index = {seq: idx for idx, seq in enumerate(self.seq_numbers_list)} # 创建序列编号到索引的映射
            print(seq_to_index)
            self.net.eval()

            Th_Seg = np.array([0.5])
            # Th_Seg = np.array(
            # [0, 1e-30, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
            #  1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2,  0.3, 0.4,  0.5,  0.6,  0.7,
            #  0.8, 0.9, 0.95, 0.99, 0.999,0.9999, 0.99999, 0.999999, 1])##置信阈值设置
            if epoch < args.epochs-1:
                Th_Seg = np.array([0, 1e-1, 0.2, 0.3, .35, 0.4, .45, 0.5, .55, 0.6, .65, 0.7, 0.8, 0.9, 0.95, 1])
            FalseNumAll = np.zeros([6,len(Th_Seg)])
            TrueNumAll = np.zeros([6,len(Th_Seg)])
            TgtNumAll = np.zeros([6,len(Th_Seg)])
            FatNumAll=np.zeros([6,len(Th_Seg)])
            pixelsNumber = np.zeros(6)
            time_start = time.time()
            low_snr3 = [6,11,18]
            high_snr3 = [20,12,15]
            time_start = time.time()
            current_seq_num = None
            image_counter = 0
            for i, data in enumerate(tqdm(self.val_loader), 0):
                Seq_num = self.seq_numbers[i]
                index = seq_to_index[Seq_num]
                if current_seq_num is None or Seq_num != current_seq_num:# 序列编号改变，重置计数器
                    current_seq_num = Seq_num
                    print(Seq_num)
                    image_counter = 0
                    OldFlag = 0
                    Old_Feat = torch.zeros([1,32,4,512,512]).to(self.device)
                with torch.no_grad():
                    SeqData_t, TgtData_t, m, n = data

                    m = int(m.float().mean().item())    #yb
                    n = int(n.float().mean().item())    #yb

                    SeqData, TgtData = Variable(SeqData_t).to(self.device), Variable(TgtData_t).to(self.device)
                    if 'DTUM' in args.model:
                        outputs,Old_Feat = run_model(0,self.net, args.model, SeqData,Old_Feat,0)
                        OldFlag=0
                    else:
                        outputs = run_model(0,self.net, args.model, SeqData,Old_Feat,OldFlag)

                    if isinstance(outputs, list):
                        outputs = outputs[0]
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    outputs = torch.squeeze(outputs, 2)

                    Outputs_Max = torch.sigmoid(outputs)
                    TestOut = Outputs_Max.data.cpu().numpy()[0, 0, 0:m, 0:n]

                    pixelsNumber[index] += m * n    #源码
                    # pixelsNumber += m * n    #yb
                    if self.save_flag:
                        img = Image.fromarray(uint8(TestOut * 255))
                        folder_name = "%sSequence%d/" % (self.test_save, Seq_num)
                        if not os.path.exists(folder_name):
                            os.mkdir(folder_name)
                        name = folder_name + ('%05d.png' % (image_counter))
                        img.save(name)
                        
                            
                        # 更新计数器
                        image_counter += 1


                    if self.writeflag:
                        for th_i in range(len(Th_Seg)):
                            TrueNum, TgtNum,Fat,FalseNum = self.eval_metrics(Outputs_Max[:,:,:m,:n], TgtData[:,:,:m,:n], Th_Seg[th_i])
                            FalseNumAll[index, th_i] = FalseNumAll[index, th_i] + FalseNum       #源码
                            TrueNumAll[index, th_i] = TrueNumAll[index, th_i] + TrueNum          #源码
                            TgtNumAll[index, th_i] = TgtNumAll[index, th_i] + TgtNum           #源码
                            FatNumAll[index, th_i] = FatNumAll[index, th_i] + Fat


            time_end = time.time()
            print('FPS=%.3f' % ((i+1)/(time_end-time_start)))

            if self.writeflag:

                Pd_lSNR = np.sum(TrueNumAll[0:3, :], axis=0) / np.sum(TgtNumAll[0:3, :], axis=0)
                Pd_hSNR = np.sum(TrueNumAll[3:, :], axis=0) / np.sum(TgtNumAll[3:, :], axis=0)
                Pd_all = np.sum(TrueNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)
                Fa_lSNR = np.sum(FalseNumAll[0:3, :], axis=0) / pixelsNumber[0:3].sum()
                Fa_hSNR = np.sum(FalseNumAll[3:, :], axis=0) / pixelsNumber[3:].sum()
                Fa_all = np.sum(FalseNumAll[:, :], axis=0) / pixelsNumber.sum()

                Fat_all= np.sum(FatNumAll[:, :], axis=0) / np.sum(TgtNumAll[:, :], axis=0)

                writelines = open(self.SavePath + 'Epoch' + str(epoch+1) + '.txt', 'w')
                for i in range(6):
                    seq = (low_snr3+high_snr3)[i]
                    writelines.write('Seq' + str(seq) + 'results:\n')
                    for seg_i in range(len(Th_Seg)):
                        writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[seg_i], TrueNumAll[i, seg_i], TgtNumAll[i, seg_i],
                                    TrueNumAll[i, seg_i] / TgtNumAll[i, seg_i], FalseNumAll[i, seg_i], FalseNumAll[i, seg_i] / pixelsNumber[i],
                                    FatNumAll[i, seg_i],TgtNumAll[i, seg_i],FatNumAll[i, seg_i] / TgtNumAll[i, seg_i]))

                writelines.write('Low SNR results:\n' )
                for th_i in range(len(Th_Seg)):
                    writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[th_i], 
                                    TrueNumAll[0:3:, th_i].sum(),TgtNumAll[0:3:, th_i].sum(),TrueNumAll[0:3:, th_i].sum() / TgtNumAll[0:3:, th_i].sum(),
                                    FalseNumAll[0:3:, th_i].sum(), FalseNumAll[0:3:, th_i].sum() / pixelsNumber[0:3].sum(),
                                    FatNumAll[0:3:, th_i].sum(),TgtNumAll[0:3:, th_i].sum(),FatNumAll[0:3:, th_i].sum() / TgtNumAll[0:3:, th_i].sum(),
                                    ))

                writelines.write('High SNR results:\n' )
                for th_i in range(len(Th_Seg)):
                    writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[th_i], 
                                    TrueNumAll[3:, th_i].sum(),TgtNumAll[3:, th_i].sum(),TrueNumAll[3:, th_i].sum() / TgtNumAll[3:, th_i].sum(),
                                    FalseNumAll[3:, th_i].sum(), FalseNumAll[3:, th_i].sum() / pixelsNumber[3:].sum(),
                                    FatNumAll[3:, th_i].sum(),TgtNumAll[3:, th_i].sum(),FatNumAll[3:, th_i].sum() / TgtNumAll[3:, th_i].sum(),
                                    ))

                writelines.write('Final results:\n' )
                for th_i in range(len(Th_Seg)):
                    writelines.write('Th_Seg = %e:\tPD:[%d/%d, %.5f]\tFA:[%d, %e]\tFAT:[%d/%d, %.5f]\n' % (Th_Seg[th_i], 
                                    TrueNumAll[:, th_i].sum(),TgtNumAll[:, th_i].sum(),TrueNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                                    FalseNumAll[:, th_i].sum(), FalseNumAll[:, th_i].sum() / pixelsNumber.sum(),
                                    FatNumAll[:, th_i].sum(),TgtNumAll[:, th_i].sum(),FatNumAll[:, th_i].sum() / TgtNumAll[:, th_i].sum(),
                                    ))
                writelines.close()

                seg = 0
                if epoch < args.epochs-1:
                    seg = 5
                print('model: %s, epoch: %d, Th_Seg = %.4e, PD:[%d, %.5f], FA:[%d, %.4e], FAT:[%d, %.5f]' % (args.model + args.loss_func, epoch + 1,
                    Th_Seg[seg], TrueNumAll[:,seg].sum(), Pd_all[seg], FalseNumAll[:,seg].sum(), Fa_all[seg], FatNumAll[:,seg].sum(),Fat_all[seg]))
    

    def savemodel(self, epoch):
        self.ModelPath, self.ParameterPath, self.SavePath = generate_savepath(self.args, epoch, self.epoch_loss)
        torch.save(self.net, self.ModelPath)
        torch.save(self.net.state_dict(), self.ParameterPath)
        print('save net OK in %s' % self.ModelPath)


    def saveloss(self):
        CurTime = time.strftime("%Y_%m_%d__%H_%M", time.localtime())
        print(CurTime)

        ###########save lost_list
        LossMatSavePath = self.SavePath + 'loss_list_' + CurTime + '.mat'
        scio.savemat(LossMatSavePath, mdict={'loss_list': self.loss_list})

        ############plot
        x1 = range(self.args.epochs)
        y1 = self.loss_list
        fig = plt.figure()
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch')
        plt.ylabel('train loss')
        LossJPGSavePath = self.SavePath + 'train_loss_' + CurTime + '.jpg'
        plt.savefig(LossJPGSavePath)
        # plt.show()
        print('finished Show!')




if __name__ == '__main__':
    args = parse_args()
    StartTime = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    print(StartTime)

    # GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    # torch.cuda.set_device(0)

    trainer = Trainer(args)
    if args.train == 1:
        for epoch in range(args.epochs):
            trainer.training(epoch)

            if (epoch+1)%10 == 0: #or epoch ==0:
                trainer.savemodel(epoch)
                trainer.validation(epoch)

        # trainer.savemodel()
        trainer.saveloss()
        print('finished training!')
        endtime = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
    if args.test == 1:
        #####################################################
        trainer.ModelPath = args.pth_path
        trainer.test_save = trainer.SavePath[0:-1] + '_visualization/'
        trainer.net = torch.load(trainer.ModelPath, map_location=trainer.device)
        print('load OK!')
        epoch = args.epochs
        #####################################################
        trainer.validation(epoch)
        endtime = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
        print(endtime)







