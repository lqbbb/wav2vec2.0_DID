import torch
import torch.nn as nn
import torch.nn.functional as F

# from libs.nnet import *
# from utils import  *


#小数据集是=时 conv_kernels=[1, 3, 5, 7], dilation=2,3,4
#大数据集是=时 conv_kernels=[ 3, 5, 7,9], dilation=2,3,4
#小数据集是=时 conv_kernels=[1, 3, 5, 7], dilation=1,2,3 (试一试这个)
#加入specaugment，不用dropout
''' Res2Conv1d + BatchNorm1d + ReLU
'''

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''
    # def __init__(self, channels,  dilation=1, scale=8):
    def __init__(self, channels, dilation=1, scale=8, kernel_size=1, stride=1, padding=0, bias=False):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            # self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.convs.append(PSAModule(self.width, self.width, dilation))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out



''' Conv1d + BatchNorm1d + ReLU
'''
class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))

#为了调用
class Conv1dReluBn_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):

        x = x.transpose(1,2)
        x = self.conv(x)
        x = self.bn(F.relu(x))
        return x.transpose(1,2)


''' The SE connection of 1D case.
'''
class SE_Connect(nn.Module):
    def __init__(self, channels, s=2):
        super().__init__()
        assert channels % s == 0, "{} % {} != 0".format(channels, s)  #把channesl改为了channels
        self.linear1 = nn.Linear(channels, channels // s)
        self.linear2 = nn.Linear(channels // s, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)
        return out


class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=2):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)
#PSA模块
class PSAModule(nn.Module):


    def __init__(self, inplans, planes, dilation,  conv_kernels=[1, 3, 5, 7], stride=1):

        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=(conv_kernels[0]//2)*dilation, dilation=dilation,
                            stride=stride )
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=(conv_kernels[1]//2)*dilation, dilation=dilation,
                            stride=stride)
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=(conv_kernels[2]//2)*dilation, dilation=dilation,
                            stride=stride)
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=(conv_kernels[3]//2)*dilation, dilation=dilation,
                            stride=stride)
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        # print("x=", x.shape)
        x1 = self.conv_1(x)
        # print("x1=",x1.shape)
        x2 = self.conv_2(x)
        # print("x2=", x2.shape)
        x3 = self.conv_3(x)
        # print("x3=", x3.shape)
        x4 = self.conv_4(x)
        # print("x4=", x4.shape)

        feats = torch.cat((x1, x2, x3, x4), dim=1)

        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        # attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1)
        # print("attention_vectors=", attention_vectors.shape)
        attention_vectors = self.softmax(attention_vectors)
        # print("attention_vectors1=", attention_vectors.shape)
        feats_weight = feats * attention_vectors
        # print(" feats_weight=",  feats_weight.shape)
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)
        return out

''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
'''
def SE_Res2Block(channels,dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels,  dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)   #之前第一版有这个
    )


#用来给conformer模块调用
def SE_Res2Block_2(channels,dilation, scale):
    return nn.Sequential(
        # Conv1dReluBn(channels,  channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn( channels, dilation, scale=scale),
        # Conv1dReluBn( channels,  channels, kernel_size=1, stride=1, padding=0),
        # SE_Connect(channels)   #之前第一版有这个
    )



''' Attentive weighted mean and standard deviation pooling.
'''
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



''' Implementation of
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification".

    Note that we DON'T concatenate the last frame-wise layer with non-weighted mean and standard deviation, 
    because it brings little improvment but significantly increases model parameters. 
    As a result, this implementation basically equals the A.2 of Table 2 in the paper.
'''
class ECAPA_TDNN(nn.Module):
    def __init__(self,num_classes=17,in_channels=83, channels=512, embd_dim=192,specaugment=True, specaugment_params={},):  #本来只有in_channels=80, channels=512, embd_dim=192
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, dilation=2,scale=8)   #scale等于8还是4  之前微调9的时候是4
        self.layer3 = SE_Res2Block(channels, dilation=3,scale=8)
        self.layer4 = SE_Res2Block(channels, dilation=4,scale=8)


        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, cat_channels, kernel_size=1)
        self.pooling = AttentiveStatsPool(cat_channels, 128)
        self.bn1 = nn.BatchNorm1d(cat_channels * 2)
        self.linear = nn.Linear(cat_channels * 2, embd_dim)
        self.bn2 = nn.BatchNorm1d(embd_dim)

        #自己加的

        default_specaugment_params = {
            "frequency": 0.05,
            "frame": 0.05,
            "rows": 2, "cols": 2,
            "random_rows": True,
            "random_cols": True
        }

        specaugment_params = utils.assign_params_dict(default_specaugment_params, specaugment_params)
        self.specaugment = SpecAugment(**specaugment_params) if specaugment else None


        self.classifier = nn.Linear(embd_dim, num_classes)  #自己加的


    # 自己加的auto

    def auto(self, layer, x):
        """It is convenient for forward-computing when layer could be None or not
        """
        return layer(x) if layer is not None else x

    def forward(self, x):

        x = x.transpose(1, 2)

        out1 = self.layer1(x)
        #ZJ
        # out1 = self.auto(self.specaugment, out1)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3
        # print("out4=",out4.shape)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        # print("out=", out.shape)
        out = self.bn2(self.linear(out))   #这应该是嵌入层

        # print("out1=", out.shape)
        # 后面自己加的

        # out = out.view(out.size(0), -1)

        y = self.classifier(out)
        #自己
        y = F.log_softmax(y, dim=-1)

        return y

#	# para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
def get_model_size(model):
    para_num = sum([p.numel() for p in model.parameters()])
    para_size = para_num / 1024 / 1024
    return para_size


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    x = torch.rand(32, 300, 83)    #torch.zeros()返回一个由标量值0填充的张量
    model = ECAPA_TDNN(in_channels=83, channels=512, embd_dim=192)
    out = model(x)
    p = get_model_size(model)
    print(p)


    # print(model)
    print(out.shape)    # should be [2, 192]
