from torch import nn
import torch
import numpy as np

class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        param loss:
        param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors

        self.loss = loss

    def forward(self, x, y ,disMap = None,epoch = None , do_disMap_ds = True , Tauo_ds = 0 , ds_epoch = 1000):

        if do_disMap_ds == True:
            assert isinstance(x, (tuple, list)), "x must be either tuple or list"
            assert isinstance(y, (tuple, list)), "y must be either tuple or list"
            assert isinstance(disMap, (tuple, list)), "y must be either tuple or list"
            if self.weight_factors is None:
                weights = [1] * len(x)
            else:
                weights = self.weight_factors

            l = weights[0] * self.loss(x[0], y[0],disMap[0],epoch)

            # smooth transition with deep supervision
            for i in range(1, len(x)):
                if weights[i] != 0:
                    if epoch <= Tauo_ds:
                        disMap[i][:, 0] = disMap[i][:, 0]

                    # Gradual blending of distance Map to uniform during smooth transition period
                    elif Tauo_ds < epoch <= Tauo_ds + ds_epoch:
                        if disMap[i][:, 0] != None:
                            temp_disMap_value = disMap[i][:, 0] / torch.mean(disMap[i][:, 0])
                            warm_start_matrix = torch.ones_like(temp_disMap_value)
                            warm_para = float(Tauo_ds + ds_epoch - epoch) / ds_epoch
                            temp_disMap_value = warm_para * warm_start_matrix + (1 - warm_para) * temp_disMap_value

                            disMap_ds_para = weights[i]
                            weighted_ds_disMap = disMap_ds_para * temp_disMap_value + (1 - disMap_ds_para) * warm_start_matrix
                            weighted_ds_disMap = weighted_ds_disMap / torch.mean(weighted_ds_disMap)
                            disMap[i][:, 0] = weighted_ds_disMap

                    elif epoch > Tauo_ds + ds_epoch:
                        # print(do_disMap_ds)
                        if disMap[i][:, 0] != None:
                            temp_disMap_value = disMap[i][:,0] / torch.mean(disMap[i][:, 0])
                            ones_matrix = torch.ones_like(temp_disMap_value)
                            disMap_ds_para = weights[i]
                            weighted_ds_disMap = disMap_ds_para * temp_disMap_value + (1 - disMap_ds_para) * ones_matrix

                            weighted_ds_disMap = weighted_ds_disMap / torch.mean(weighted_ds_disMap)
                            disMap[i][:, 0] = weighted_ds_disMap
                        else:
                            print("We don't use Distance Map weighted loss")
                    l += weights[i] * self.loss(x[i], y[i],disMap[i],epoch)

            return l

        else:
            assert isinstance(x, (tuple, list)), "x must be either tuple or list"
            assert isinstance(y, (tuple, list)), "y must be either tuple or list"
            if self.weight_factors is None:
                weights = [1] * len(x)
            else:
                weights = self.weight_factors

            l = weights[0] * self.loss(x[0], y[0])
            for i in range(1, len(x)):
                if weights[i] != 0:
                    l += weights[i] * self.loss(x[i], y[i])
            return l
