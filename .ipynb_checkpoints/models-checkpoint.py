import torch
import torch.nn as nn
import torch.nn.functional as F
import math



        
class CNN1D(nn.Module):
    def __init__(self, output_size):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
        )
        self.avgpool = nn.AdaptiveAvgPool1d(64) # output (batch, 64, 64)
        self.fc = nn.Linear(64*64, output_size)

    def forward(self, x, return_features=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)  # [batch, 64, 64]
        features = x.view(x.size(0), -1)  # [batch, 64*64]
        
        if return_features:
            return features
        
        x = self.fc(features)
        return x

class tongdao(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        y = self.relu(y)
        y = nn.functional.interpolate(y, size=(x.size(2),), mode='nearest')
        return x * y.expand_as(x)

class kongjian(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv1d(in_channel, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, x):
        y = self.Conv1x1(x)
        y = self.norm(y)
        return x * y

class hebing(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.tongdao = tongdao(in_channel)
        self.kongjian = kongjian(in_channel)

    def forward(self, U):
        U_kongjian = self.kongjian(U)
        U_tongdao = self.tongdao(U)
        return torch.max(U_tongdao, U_kongjian)

class MRFEN(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(MRFEN, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm1d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm1d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm1d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm1d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm1d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv1d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm1d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.Hebing = hebing(in_channel=dim_out*5)

    def forward(self, x):
        [b, c, length] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        global_feature = torch.mean(x, 2, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (length,), mode='nearest')
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        larry = self.Hebing(feature_cat)
        larry_feature_cat = larry * feature_cat
        result = self.conv_cat(larry_feature_cat)
        return result



class CSSA(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(CSSA, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm1d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(in_channels)
        )

    def channel_shuffle(self, x, groups):
        batchsize, num_channels, length = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, length)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, length)
        return x

    def forward(self, x):
        b, c, l = x.shape
        x_permute = x.permute(0, 2, 1).contiguous()
        x_att_permute = self.channel_attention(x_permute).view(b, l, c)
        x_channel_att = x_att_permute.permute(0, 2, 1).sigmoid()
        x = x * x_channel_att
        x = self.channel_shuffle(x, groups=4)
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att
        return out

class CNN1D_MRFEN_CSSA(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=10, rate=1, bn_mom=0.1):
        super(CNN1D_MRFEN_CSSA, self).__init__()
        
        # CNN1D部分
        self.cnn1d = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size*2, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # MRFEN
        self.MRFEN_branch1 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.MRFEN_branch2 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.MRFEN_branch3 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.MRFEN_branch4 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # CSSA
        self.CSSA = CSSA(hidden_size*16)  # 4个分支，每个分支hidden_size*4
        
        # 最终分类部分
        self.conv_cat = nn.Sequential(
            nn.Conv1d(hidden_size*16, hidden_size*4, 1, 1, padding=0, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size*4, output_size)

    def forward(self, x, return_features=False):
        # CNN1D
        x = self.cnn1d(x)
        
        # MRFEN
        x1 = self.MRFEN_branch1(x)
        x2 = self.MRFEN_branch2(x)
        x3 = self.MRFEN_branch3(x)
        x4 = self.MRFEN_branch4(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        
        # CSSA
        x_CSSA = self.CSSA(x_cat)
        
        # 提取特征
        features = self.conv_cat(x_CSSA)
        features = self.gap(features).squeeze(-1)
        
        if return_features:
            return features
        
        # 最终分类
        output = self.fc(features)
        
        return output

class CNN1D_MRFEN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=10, rate=1, bn_mom=0.1):
        super(CNN1D_MRFEN, self).__init__()
        
        # CNN1D部分
        self.cnn1d = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size*2, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # MRFEN
        self.MRFEN_branch1 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.MRFEN_branch2 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.MRFEN_branch3 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.MRFEN_branch4 = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # 最终分类部分
        self.conv_cat = nn.Sequential(
            nn.Conv1d(hidden_size*16, hidden_size*4, 1, 1, padding=0, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size*4, output_size)

    def forward(self, x, return_features=False):
        # CNN1D
        x = self.cnn1d(x)
        
        # MRFEN
        x1 = self.MRFEN_branch1(x)
        x2 = self.MRFEN_branch2(x)
        x3 = self.MRFEN_branch3(x)
        x4 = self.MRFEN_branch4(x)
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)
        
        # 提取特征
        features = self.conv_cat(x_cat)
        features = self.gap(features).squeeze(-1)
        
        if return_features:
            return features
        
        # 最终分类
        output = self.fc(features)
        
        return output


class CNN1D_CSSA(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=10, bn_mom=0.1):
        super(CNN1D_CSSA, self).__init__()
        
        # CNN1D部分
        self.cnn1d = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size*2, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # CSSA
        self.CSSA = CSSA(hidden_size*4)
        
        # 最终分类部分
        self.conv_cat = nn.Sequential(
            nn.Conv1d(hidden_size*4, hidden_size*4, 1, 1, padding=0, bias=True),
            nn.BatchNorm1d(hidden_size*4, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size*4, output_size)

    def forward(self, x, return_features=False):
        # CNN1D
        x = self.cnn1d(x)
        
        # CSSA
        x_CSSA = self.CSSA(x)
        
        # 提取特征
        features = self.conv_cat(x_CSSA)
        features = self.gap(features).squeeze(-1)
        
        if return_features:
            return features
        
        # 最终分类
        output = self.fc(features)
        
        return output