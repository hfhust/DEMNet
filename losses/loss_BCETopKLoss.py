import torch
import torch.nn as nn
import torch.nn.functional as F


class MyWeightBCETopKLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False, MaxClutterNum=39, ProtectedArea=2):
        super(MyWeightBCETopKLoss, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(reduce=False)
        self.HardRatio = 1 / 4
        self.HardNum = round(MaxClutterNum * self.HardRatio)
        self.EasyNum = MaxClutterNum - self.HardNum
        self.MaxClutterNum = MaxClutterNum
        self.ProtectedArea = ProtectedArea
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):  # Input: [B, C, H, W]    Target: [B, C, H, W]
        if input.dim() > 4:
            input = torch.squeeze(input, 2)

        # Get the actual dimensions of the input and target
        B, C, H, W = input.size()

        # Create the template for the protected area
        template = torch.ones(1, 1, 2 * self.ProtectedArea + 1, 2 * self.ProtectedArea + 1).to(input.device)
        target_prot = F.conv2d(target.float(), template, stride=1, padding=self.ProtectedArea)
        target_prot = (target_prot > 0).float()

        with torch.no_grad():
            loss_wise = self.bce_loss(input, target.float())
            loss_p = loss_wise * (1 - target_prot)
            
            # Calculate the total number of elements in the flattened loss tensor
            total_elements = H * W
            
            # Generate a random permutation of indices
            idx = torch.randperm(total_elements) + 20  # Assuming 20 is a fixed offset
            
            batch_l = loss_p.shape[0]
            Wgt = torch.zeros_like(loss_p)
            for ls in range(batch_l):
                loss_ls = loss_p[ls, :, :, :].reshape(-1)
                
                # Select the top k losses
                k = min(200, total_elements)  # Ensure k does not exceed the total number of elements
                loss_topk, indices = torch.topk(loss_ls, k)
                
                # Randomly select HardNum samples from the top k losses
                indices_rand = indices[idx[0:self.HardNum]]
                
                # Randomly select EasyNum samples from all image
                idx_easy = torch.randperm(total_elements)[:self.EasyNum].to(input.device)
                
                # Combine the selected indices
                indices_rand = torch.cat((indices_rand, idx_easy), 0)
                
                # Convert the 1D indices to 2D indices
                indices_rand_row = indices_rand // W
                indices_rand_col = indices_rand % W
                
                # Set the weights
                Wgt[ls, 0, indices_rand_row, indices_rand_col] = 1

            # Update the weight data
            WgtData_New = Wgt * (1 - target_prot) + target.float()
            WgtData_New[WgtData_New > 1] = 1

        # Compute the focal loss
        logpt = F.logsigmoid(input)
        logpt_bk = F.logsigmoid(-input)
        pt = logpt.data.exp()
        pt_bk = 1 - logpt_bk.data.exp()
        loss = -self.alpha[1] * (1 - pt) ** self.gamma * target * logpt - self.alpha[0] * pt_bk ** self.gamma * (1 - target) * logpt_bk

        # Apply the weights
        loss = loss * WgtData_New

        return loss.sum()

        # if input.dim()>2:
        #     input=input.view(input.size(0), input.size(1),-1)   # N,C,D,H,W=>N,C,D*H*W
        #     input=input.transpose(1,2)                          # N,C,D*H*W=>N, D*H*W, C
        #     input=input.contiguous().view(-1,input.size(2))     # N,D*H*W,C=>N*D*H*W, C
        #
        #     WgtData_New = WgtData_New.view(WgtData_New.size(0), WgtData_New.size(1), -1)    # N,C,D,H,W=>N,C,D*H*W
        #     WgtData_New = WgtData_New.transpose(1, 2)                               # N,C,D*H*W=>N, D*H*W,C
        #     WgtData_New = WgtData_New.contiguous().view(-1, WgtData_New.size(2))        # N,D*H*W,C=>N*D*H*W,C
        #
        # target = target.view(-1,1)     ## [2*1*512*512,1]     #N,D,H,W=>1,N*D*H*W
        # logpt = F.log_softmax(input, dim=1)   ## [2*1*512*512,2]
        # logpt = logpt.gather(1,target) ##  zhiding rank 2 target
        # logpt = logpt*WgtData_New        #weight  ## predit of concern 39+1
        # logpt = logpt.view(-1)           #possibility of target
        # pt=logpt.data.exp()
        #
        #
        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha=self.alpha.type_as(input.data).to(input.device)
        #     at=self.alpha.gather(0,target.data.view(-1))
        #     logpt=logpt*at   ##at= alpha
        #
        # loss=-1*(1-pt)**self.gamma*logpt


