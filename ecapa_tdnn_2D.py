import torch
import torch.nn as nn
import torch.nn.functional as F


#加入specaugment，不用dropout
''' Res2Conv2d + BatchNorm2d + ReLU
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Res2Conv2dReluBn(nn.Module):
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
            self.convs.append(nn.Conv2d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm2d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

        # self.selu = nn.SELU(inplace=True)

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
            # sp = self.selu(self.bns[i](sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)
        return out



''' Conv2d + BatchNorm2d + ReLU
'''
class Conv2dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        # return self.selu(self.bn(self.conv(x)))
        return self.bn(F.relu(self.conv(x)))



''' The SE connection of 1D case.
'''
# class SE_Connect(nn.Module):
#     def __init__(self, channels, s=2):
#         super().__init__()
#         assert channels % s == 0, "{} % {} != 0".format(channels, s)  #把channesl改为了channels
#         self.linear1 = nn.Linear(channels, channels // s)
#         self.linear2 = nn.Linear(channels // s, channels)
#
#     def forward(self, x):
#         out = x.mean(dim=2)
#         out = F.relu(self.linear1(out))
#         out = torch.sigmoid(self.linear2(out))
#         out = x * out.unsqueeze(2)
#         return out


#2D SE
class SE_Connect(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        # 空间信息进行压缩
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 经过两次全连接层，学习不同通道的重要性
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio, False),
            nn.ReLU(),
            nn.Linear(channels // ratio, channels, False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 取出batch size和通道数
        # b,c,w,h->b,c,1,1->b,c 压缩与通道信息学习
        avg = self.avgpool(x).view(b, c)
        # b,c->b,c->b,c,1,1 激励操作
        y = self.fc(avg).view(b, c, 1, 1)
        return x * y.expand_as(x)





''' SE-Res2Block.
    Note: residual connection is implemented in the ECAPA_TDNN model, not here.
'''
def SE_Res2Block(channels, kernel_size, stride, padding, dilation, scale):
    return nn.Sequential(
        Conv2dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        Res2Conv2dReluBn(channels, kernel_size, stride, padding, dilation, scale=scale),
        Conv2dReluBn(channels, channels, kernel_size=1, stride=1, padding=0),
        SE_Connect(channels,channels)
    )



#padding = (kernel_size // 2) * dilation,
class ECAPA_TDNN(nn.Module):
    def __init__(self,in_channels=1,channels=64):  #本来只有in_channels=80, channels=512, embd_dim=192
        super().__init__()
        self.layer1 = Conv2dReluBn(in_channels, channels, kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=2, dilation=2, scale=8)  #之前是8
        self.layer3 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=3, dilation=3, scale=8)
        # self.layer3 = SE_Res2Block(channels, kernel_size=5, stride=1, padding=6, dilation=3, scale=8)
        self.layer4 = SE_Res2Block(channels, kernel_size=3, stride=1, padding=4, dilation=4, scale=8)
        # self.layer4 = SE_Res2Block(channels, kernel_size=7, stride=1, padding=12, dilation=4, scale=8)  #改了卷积核  K



    def forward(self, x):

        # x = x.transpose(1, 2)
        out1 = self.layer1(x)

        out2 = self.layer2(out1) + out1
        out3 = self.layer3(out1 + out2) + out1 + out2
        out4 = self.layer4(out1 + out2 + out3) + out1 + out2 + out3

        out = torch.cat([out2, out3, out4], dim=1)

        return out




if __name__ == '__main__':
    # Input size: batch_size * seq_len * feat_dim
    x = torch.rand(4,32,42,67)    #torch.zeros()返回一个由标量值0填充的张量
    model = ECAPA_TDNN(in_channels=32, channels=64)
    print(model)
    out = model(x)
    print(out.shape)    # should be [2, 192]
