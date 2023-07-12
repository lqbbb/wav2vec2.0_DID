import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
from RawNet3_no_sinc import MainModel

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

############################
## FOR fine-tuned SSL MODEL
############################

#直接接平均池化,或者注意力池户
class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        # cp_path = '/medias/speech/projects/tak/our_staging/LA/Baseline-RawNet2/pre-trained_model_SSL/models/XLR_300M/xlsr2_300m.pt'   # change the pre-trained XLSR model path as per your directoy
        cp_path = '/home/lqb/project/SSL_Anti-spoofing-main/pre-trained_model_SSL/xlsr2_300m.pt'   # change the pre-trained XLSR model path as per your directoy
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there  将模型放入 GPU（如果它不存在）
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            # input should be in shape (batch, length)  输入应为形状（批次、长度）
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb
#自己
class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1) # equals W and b in the paper
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1) # equals V and k in the paper

    def forward(self, x):
        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * x ** 2, dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class Model(nn.Module):
    def __init__(self,args,device):
    # def __init__(self,device):
        super().__init__()
        self.device=device

        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)

        self.pooling = AttentiveStatsPool(128, 64)
        self.bn1 = nn.BatchNorm1d(256)
        #自己
        self.classifier = nn.Linear(128,4)  #平均池化用
        # self.classifier = nn.Linear(256,4)  #注意力池化用


    def forward(self, x, Freq_aug=False):

        x_ssl_feat = self.ssl_model.extract_feat(x.squeeze(-1))
        x=self.LL(x_ssl_feat) #(bs,frame_number,feat_out_dim)


        # 自己
        x = x.transpose(1, 2)  # (bs,feat_out_dim,frame_number)
        x = x.mean(dim=-1)

        #注意力池化
        # x = self.bn1(self.pooling(x)) #应该没有做




        output = self.classifier(x)


        return output


if __name__ == '__main__':
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # device = 'cuda'
    print('Device: {}'.format(device))

    model = Model(device).to(device)
    # print(model)
    wav = torch.randn(2, 64600).to(device)
    # print(wav.shape)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    batch_out = model(wav)
    print(batch_out)

