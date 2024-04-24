import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    A custom loss function that extends PyTorch's CrossEntropyLoss to incorporate a dynamic modulation based on a
    distance map and epoch-based smoothing.
    """
    def forward(self, input: Tensor, target: Tensor, disMap=None, current_epoch=None) -> Tensor:
        Tauo_st = 0 # Start epoch for smooth transition
        st_epoch = 1000 # Duration of the smooth transition
        smooth_trans = True
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
            if disMap != None:
                disMap = disMap[:, 0]

        # Distance map weighted loss without Smooth transition
        if smooth_trans == False:
            temp = super().forward(input, target.long())
            if disMap != None:
                disMap = disMap / torch.mean(disMap)
                temp_disMap_value = disMap
                if temp_disMap_value.shape == temp.shape:
                    temp_after_DisMap = torch.mul(temp, temp_disMap_value)
            else:
                temp_after_DisMap = temp
            return temp_after_DisMap

        # Distance map weighted loss with Smooth transition
        if smooth_trans == True:
            temp = super().forward(input, target.long())
            if disMap != None:
                disMap = disMap / torch.mean(disMap)
                temp_disMap_value = disMap
                if current_epoch < Tauo_st:
                    temp_after_DisMap = temp
                elif Tauo_st <= current_epoch < Tauo_st + st_epoch:
                    if temp_disMap_value.shape == temp.shape:
                        warm_start_matrix = torch.ones_like(temp_disMap_value)
                        warm_para = float(Tauo_st + st_epoch - current_epoch) / st_epoch
                        temp_disMap_value_w = warm_para * warm_start_matrix + (1 - warm_para) * temp_disMap_value

                        temp_after_DisMap = torch.mul(temp, temp_disMap_value_w)
                elif current_epoch >= Tauo_st + st_epoch:
                    if temp_disMap_value.shape == temp.shape:
                        temp_after_DisMap = torch.mul(temp, temp_disMap_value)
            else:
                temp_after_DisMap = temp
            return temp_after_DisMap