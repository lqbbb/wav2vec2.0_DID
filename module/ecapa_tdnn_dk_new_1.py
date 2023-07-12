import torch
import torch.nn as nn
import torch.nn.functional as F

# from libs.nnet import *
# from utils import  *
import torch.utils.checkpoint as cp
#不加入specaugment，不用dropout
#不使用高阶统计模型
def get_nonlinear(config_str, channels):
    nonlinear = nn.Sequential()
    for name in config_str.split('-'):
        if name == 'relu':
            nonlinear.add_module('relu', nn.ReLU(inplace=True))
        elif name == 'prelu':
            nonlinear.add_module('prelu', nn.PReLU(channels))
        elif name == 'batchnorm':
            nonlinear.add_module('batchnorm', nn.BatchNorm1d(channels))
        elif name == 'batchnorm_':
            nonlinear.add_module('batchnorm',
                                 nn.BatchNorm1d(channels, affine=False))
        else:
            raise ValueError('Unexpected module ({}).'.format(name))
    return nonlinear

def high_order_statistics_pooling(x,
                                  dim=-1,
                                  keepdim=False,
                                  unbiased=True,
                                  eps=1e-2):
    mean = x.mean(dim=dim)
    std = x.std(dim=dim, unbiased=unbiased)
    # norm = (x - mean.unsqueeze(dim=dim)) \
    #     / std.clamp(min=eps).unsqueeze(dim=dim)
    # skewness = norm.pow(3).mean(dim=dim)
    # kurtosis = norm.pow(4).mean(dim=dim)
    # stats = torch.cat([mean, std, skewness, kurtosis], dim=-1)
    stats = torch.cat([mean, std], dim=-1)
    if keepdim:
        stats = stats.unsqueeze(dim=dim)
    return stats


class HighOrderStatsPool(nn.Module):
    def forward(self, x):
        return high_order_statistics_pooling(x)



class StatsSelect(nn.Module):
    def __init__(self, channels, branches, null=False, reduction=1):
        super(StatsSelect, self).__init__()
        self.gather = HighOrderStatsPool()
        # self.linear1 = nn.Conv1d(channels * 4, channels // reduction, 1)
        self.linear1 = nn.Conv1d(channels * 2, channels // reduction, 1)
        self.linear2 = nn.ModuleList()
        if null:
            branches += 1
        for _ in range(branches):
            self.linear2.append(nn.Conv1d(channels // reduction, channels, 1))
        self.channels = channels
        self.branches = branches
        self.null = null
        self.reduction = reduction

    def forward(self, x):
        # print("x=",x.shape)
        f = torch.cat([_x.unsqueeze(dim=1) for _x in x], dim=1)
        # print("f=", f.shape)  #f= torch.Size([4, 2, 128, 300])
        x = torch.sum(f, dim=1)
        # print("x1=", x.shape)  #x1= torch.Size([4, 128, 300])
        x = self.linear1(self.gather(x).unsqueeze(dim=-1))
        # print("x2=", x.shape)  #x2= torch.Size([4, 64, 1])
        s = []
        for linear in self.linear2:
            s.append(linear(x).view(-1, 1, self.channels))
        s = torch.cat(s, dim=1)
        # print("s=", s.shape)   #s= torch.Size([4, 2, 128])
        s = F.softmax(s, dim=1).unsqueeze(dim=-1)
        # print("s1=", s.shape)  #s1= torch.Size([4, 2, 128, 1])
        if self.null:
            s = s[:, :-1, :, :]
        # print("torch.sum(f * s, dim=1)=", (torch.sum(f * s, dim=1)).shape)  #torch.sum(f * s, dim=1)= torch.Size([4, 128, 300])
        return torch.sum(f * s, dim=1)

    def extra_repr(self):
        return 'channels={}, branches={}, reduction={}'.format(
            self.channels, self.branches, self.reduction)


class DK(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=(1,),
                 bias=False,
                 null=False,
                 reduction=1):
        super(DK, self).__init__()
        assert kernel_size % 2 == 1, 'Expect equal paddings, but got even kernel size ({})'.format(
            kernel_size)
        padding = (kernel_size - 1) // 2
        if not isinstance(dilation, (tuple, list)):
            dilation = (dilation, )

        self.linear2 = nn.ModuleList()
        for _dilation in dilation:
            self.linear2.append(
                nn.Conv1d(in_channels,
                          out_channels,
                          kernel_size,
                          stride=stride,
                          padding=padding * _dilation,
                          dilation=_dilation,
                          bias=bias))
        self.select = StatsSelect(out_channels,
                                  len(dilation),
                                  null=null,
                                  reduction=reduction)

    def forward(self, x):

        x = self.select([linear(x) for linear in self.linear2])
        return x



''' Res2Conv1d + BatchNorm1d + ReLU
'''
class Res2Conv1dReluBn_DK(nn.Module):
    '''
    in_channels == out_channels == channels
    '''
    def __init__(self, channels, kernel_size=1, stride=1, dilation= (1,), bias=False, scale=4,null = False,reduction = 1):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        self.se = []
        for i in range(self.nums):
            # self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.convs.append(DK(self.width, self.width, kernel_size, stride, dilation= dilation , bias=bias,null = null,reduction = reduction))
            self.bns.append(nn.BatchNorm1d(self.width))
            # 自己想加个通道注意力模块
            self.se.append(SE_Connect(self.width))   #之前有的
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.se = nn.ModuleList(self.se)

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
            sp = self.se[i](sp)
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out

class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
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

#用来调用conformer_cat
def SE_Res2Block_DK_1(channels, kernel_size, stride,  dilation, scale, null , reduction ):
    return nn.Sequential(
        # Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn_DK(channels, kernel_size, stride, dilation, scale=scale,null=null,reduction = reduction),
        # Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        # SE_Connect(channels)
    )

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



''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
'''

def SE_Res2Block_DK(channels, kernel_size, stride,  dilation, scale, null , reduction ):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn_DK(channels, kernel_size, stride,  dilation, scale=scale,null=null,reduction = reduction),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
    )

def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv1dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv1dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels)
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
    # def __init__(self,num_classes=17,in_channels=83, channels=1024, embd_dim=192,specaugment=True, specaugment_params={},):  #本来只有in_channels=80, channels=512, embd_dim=192
        super().__init__()
        self.layer1 = Conv1dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block_DK(channels, kernel_size=3, stride=1, dilation=(1,2), scale=8,null=False,reduction = 2)  #之前小数据是4
        self.layer3 = SE_Res2Block_DK(channels, kernel_size=5, stride=1, dilation=(1,3), scale=8,null=False,reduction = 2)
        self.layer4 = SE_Res2Block_DK(channels, kernel_size=7, stride=1, dilation=(1,4), scale=8,null=False,reduction = 2)



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




        self.classifier = nn.Linear(embd_dim, num_classes)  #自己加的


    # 自己加的auto

    def auto(self, layer, x):
        """It is convenient for forward-computing when layer could be None or not
        """
        return layer(x) if layer is not None else x

    def forward(self, x):

        x = x.transpose(1, 2)

        out1 = self.layer1(x)
        #ZJ 放在第二层
        # out1 = self.auto(self.specaugment, out1)
        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn1(self.pooling(out))
        out = self.bn2(self.linear(out))   #这应该是嵌入层

        # 后面自己加的

        # out = out.view(out.size(0), -1)

        y = self.classifier(out)
        #自己
        y = F.log_softmax(y, dim=-1)

        return y


#	# para_size: 参数个数 * 每个4字节(float32) / 1024 / 1024，单位为 MB
def get_model_size(model):
    para_num = sum([p.numel() for p in model.parameters()])
    para_size = para_num  / 1024 / 1024  #参数
    return para_size


if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    x = torch.rand(32, 300, 83)    #torch.zeros()返回一个由标量值0填充的张量
    model = ECAPA_TDNN(in_channels=83, channels=1024, embd_dim=192)
    out = model(x)
    p = get_model_size(model)
    print(p)

    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    # print(model)
    print(out.shape)    # should be [2, 192]
