import torch
from torch import nn
from torch.nn import init
from torch.optim import lr_scheduler
import os
# from model import fuse
from torch.nn import functional as F
import numpy as np

def get_scheduler(optimizer, opt):  # 学习率调整
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)  # 自定义调整
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_gamma)  # 等间隔调整学习率
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                   factor=opt.lr_decay_gamma,
                                                   patience=opt.lr_decay_patience,
                                                   min_lr=1e-7)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):  # 权重初始化
    def init_func(m):
        classname = m.__class__.__name__  # hasattr判断m这个对象是否包含weight属性,find返回找到下标
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'mean_space':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == 'mean_channel':
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                # print('initialization method [%s] is not implemented' % init_type)
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # 会递归调用module中的参数进行初始化


def init_net(net, init_type='normal', init_gain=0.1):
    assert (torch.cuda.is_available())
    net = net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net



def define_WCNET(input_ch, init_type='normal', init_gain=0.1):
    net = WCNET(channel=input_ch)
    return init_net(net, init_type, init_gain)

def define_Prediction_Net(input_ch, init_type='normal', init_gain=0.1):
    net = Prediction_Net(channel=input_ch)
    return init_net(net, init_type, init_gain)

def define_DoubleConv(input_ch,output_ch, init_type='normal', init_gain=0.02):
    net = DoubleConv(in_ch=input_ch, out_ch=output_ch)
    return init_net(net, init_type, init_gain)

def define_Coder(input_ch, output_ch, init_type='normal', init_gain=0.02):
    net = Coder(in_ch=input_ch, out_ch=output_ch)
    return init_net(net, init_type, init_gain)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 添加了BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


##Coder
class Coder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Coder, self).__init__()
        self.conv1 = DoubleConv(1, in_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(in_ch, 2*in_ch)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(2*in_ch, 4*in_ch)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(4*in_ch, 8*in_ch)

        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up5 = nn.ConvTranspose2d(8*out_ch, 4*out_ch, 2, stride=2)
        self.conv5 = DoubleConv(8*out_ch, 4*out_ch)
        self.up6 = nn.ConvTranspose2d(4*out_ch, 2*out_ch, 2, stride=2)
        self.conv6 = DoubleConv(4*out_ch, 2*out_ch)
        self.up7 = nn.ConvTranspose2d(2*out_ch, out_ch, 2, stride=2)
        self.conv7 = DoubleConv(2*out_ch, out_ch)
        self.conv8 = nn.Conv2d(out_ch, out_ch, 1)

        self.up=nn.ConvTranspose2d(8*out_ch, out_ch, 8, stride=8)    #对中间进行特征对齐然后提取


    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)     #后面要提取

        c_plus = self.up(c4)


        up_5 = self.up5(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        c8 = self.conv8(c7)
        out = c_plus + c8


        return out

class Prediction_Net(nn.Module):
    def __init__(self, channel):
        super(Prediction_Net, self).__init__()
        self.A1 = nn.Sequential(
            nn.Conv2d(2*channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.MaxPool2d(8)
        )
        self.time_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(8)
        )

        self.A2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(8)
        )
        self.A3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(8)
        )
        self.A4 = nn.Sequential(
            nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(8)
        )
        self.A5 = nn.Sequential(
            nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(8)
        )
        self.DownConv = nn.Sequential(
            nn.Conv2d(2*channel, 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.FC1 = nn.Linear(1800, 180, bias=False)
        self.FC2 = nn.Linear(180, 1, bias=False)
        self.FC3 = nn.Linear(3, 1, bias=False)
        self.FC4 = nn.Linear(1, 1, bias=False)
        self.Conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
    def forward(self, x, x_pre, x_cloud_band, translational_speed, lon, GLCM, time_series ):
        # x1 = torch.cat([x, x_cloud_band], dim=1)
        x2 = torch.cat([x, x_pre], dim=1)
        x3 = self.A1(x2)
        batch, channel, height, width = x3.shape
        time = self.time_conv(time_series)

        y1 = self.A2(x_pre)
        y2 = self.A3(x_pre)

        sim1 = torch.cosine_similarity(y1 / height, y2, dim=1)
        sim1 = sim1.unsqueeze(1)
        sim1 = sim1.permute(0, 1, 3, 2).contiguous()

        z1 = torch.matmul(x3, sim1)
        z1 = torch.cat([z1, time], dim=1)
        z2 = self.DownConv(z1)
        z2 = z2.view(-1, 2*30*30)
        z2 = self.FC1(z2)
        z2 = self.FC2(z2)

        lx = torch.cat([z2, lon], dim=1)  #纬度

        tl = torch.cat([lx, translational_speed], dim=1)
        # GTL = torch.cat([tl, GLCM], dim=1)

        GTL = tl.view(-1, 3)
        speed1 = self.FC3(GTL)

        speed2 = speed1.unsqueeze(1)

        speed3 = self.Conv1d(speed2)
        speed4 = speed3.view(-1, 1*1)
        speed5 = self.FC4(speed4)
        return speed5, speed1, speed4

class WCNET(nn.Module):
    def __init__(self, channel):
        super(WCNET, self).__init__()
        # Cross fusion
        self.F1_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        )
        self.F1_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.F1_3 = nn.Sequential(
            nn.Conv2d(channel, 2*channel, kernel_size=1, stride=1, padding=0)
        )
        self.F1_4 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.F1_5 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.F2_1 = nn.Sequential(
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1)
        )
        self.F2_2 = nn.Sequential(
            nn.Conv2d(7, channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.F2_3 = nn.Sequential(
            nn.Conv2d(channel, 2*channel, kernel_size=1, stride=1, padding=0)
        )
        self.F2_4 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.F2_5 = nn.Sequential(
            nn.Conv2d(2*channel, 4*channel, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(2)
        )
        self.F3_1 = nn.Sequential(
            nn.ConvTranspose2d(8*channel, 4*channel, 2, stride=2)
        )
        self.F3_2 = nn.Sequential(
            nn.ConvTranspose2d(4*channel, 2*channel, 2, stride=2),
            nn.MaxPool2d(1)
        )
        self.F3_3 = nn.Sequential(
            nn.Conv2d(2*channel, channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.C1 = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
        )
        self.C2 = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0)
        )
        #1
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        )
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU()
        )

        self.BN = nn.Sequential(
            nn.BatchNorm2d(channel)
        )
        self.coder = define_Coder(channel, channel)
        self.conv_channl = DoubleConv(2*channel, channel)

    def spatial_edge_0_7(self, x):
        edge1 = x[:, 0:7, 0:x.size(2) - 1, :] - x[:, 0:7, 1:x.size(2), :]
        edge2 = x[:, 0:7, :, 0:x.size(3) - 1] - x[:, 0:7, :, 1:x.size(3)]
        return edge1, edge2

    def spatial_edge_8_14(self, x):
        edge1 = x[:, 7:, 0:x.size(2) - 1, :] - x[:, 7:, 1:x.size(2), :]
        edge2 = x[:, 7:, :, 0:x.size(3) - 1] - x[:, 7:, :, 1:x.size(3)]
        return edge1, edge2

    def spectral_edge_0_7(self,x):
        edge = x[:, 0:6, :, :] - x[:, 1:7, :, :]
        return edge

    def spectral_edge_8_14(self,x):
        edge = x[:, 7:12, :, :] - x[:, 8:13, :, :]
        return edge

    def forward(self, x_2km, x_4km):

        #cross fusion
        x_4km_1 = self.F1_1(x_4km)
        x_4km_2 = self.F1_2(x_4km_1)
        x_4km_3 = self.F1_3(x_4km_2)
        x_4km_4 = self.F1_4(x_4km_3)
        x_4km_0 = self.F1_5(x_4km_3)

        x_2km_1 = self.F2_1(x_2km)
        x_2km_2 = self.F2_2(x_2km_1)
        x_2km_3 = self.F2_3(x_2km_2)
        x_2km_4 = self.F2_4(x_2km_3)
        x_2km_0 = self.F2_5(x_2km_3)

        x_0_1 = x_4km_4 + x_2km_0
        x_0_2 = x_2km_4 + x_4km_0

        x0 = torch.cat([x_0_1, x_0_2], dim=1)
        x0_1 = self.F3_1(x0)
        x0_2 = self.F3_2(x0_1)
        x0 = self.F3_3(x0_2)

        #Cloud Band
        x_Cloud_Band = x_2km[:, 6, :, :]
        x_Cloud_Band = x_Cloud_Band.unsqueeze(1)

        x_Cloud_Band = F.interpolate(x_Cloud_Band, (60, 60), mode='bilinear', align_corners=True)
        x_Cloud_Band[:, :, 14:46, 14:46] = 0            #中间32*32为眼区
        x_Cloud_Band_1_Import = self.C1(x_Cloud_Band)          #要提到后面去融合的参数
        x_Cloud_Band_2 = self.C2(x_Cloud_Band)
        batch_Cloud_Band, channel_Cloud_Band, height_Cloud_Band, width_Cloud_Band = x_Cloud_Band_1_Import.shape
        x_Cloud_Band_1 = x_Cloud_Band_1_Import.permute(0, 2, 3, 1).contiguous()
        x_Cloud_Band_1 = x_Cloud_Band_1.reshape(-1, height_Cloud_Band * width_Cloud_Band, channel_Cloud_Band)
        x_Cloud_Band_2 = x_Cloud_Band_2.permute(0, 2, 3, 1).contiguous()
        x_Cloud_Band_2 = x_Cloud_Band_2.reshape(-1, height_Cloud_Band * width_Cloud_Band, channel_Cloud_Band)
        x_Cloud_Band_2T = torch.transpose(x_Cloud_Band_2, dim0=2, dim1=1).contiguous()
        x_Cloud_Band = torch.matmul(x_Cloud_Band_1, x_Cloud_Band_2T) / height_Cloud_Band
        x_Cloud_Band = F.softmax(x_Cloud_Band, dim=2) #云带权重

        #Eye
        x_Eyes = x_2km[:, 6, :, :]
        x_Eyes = x_Eyes.unsqueeze(1)
        x_Eyes = F.interpolate(x_Eyes, (60, 60), mode='bilinear', align_corners=True)
        x_Eye = x_Eyes[:, :, 14:46, 14:46]

        x_Eye = self.coder(x_Eye)
        batch_Eye, channel_Eye, height_Eye, width_Eye = x_Eye.shape
        x_Eye_1 = x_Eye.permute(0, 2, 3, 1).contiguous()
        x_Eye_1 = x_Eye_1.reshape(-1, height_Eye * width_Eye, channel_Eye)
        x_Eye_1T = torch.transpose(x_Eye_1, dim0=2, dim1=1).contiguous()
        x_Eye_1 = torch.matmul(x_Eye_1, x_Eye_1T) / height_Eye
        x_Eye_1 = F.softmax(x_Eye_1, dim=2) #眼区的权重


        x1_1 = F.interpolate(x0, (60, 60), mode='bilinear', align_corners=True)
        x1_1_1 = x1_1.permute(0, 2, 3, 1).contiguous()                                          #云带特征融合
        x1_1_2 = x1_1_1.reshape(-1, height_Cloud_Band * width_Cloud_Band, channel_Cloud_Band)   #b*hw*c
        x1_1_3 = torch.matmul(x_Cloud_Band, x1_1_2)                                           #b*hw*hw
        # x1_1_3 = x1_1.mul(x_Cloud_Band)
        x1_1_3 = x1_1_3.reshape(-1, height_Cloud_Band, width_Cloud_Band, channel_Cloud_Band)    #b*h*w*c
        x1_1_3 = x1_1_3.permute(0, 3, 1, 2).contiguous()                                        #b*c*h*w
        x1_1 = x1_1_3 + x1_1

        x1_2 = self.conv1_1(x1_1)
        x1_2 = self.conv1_2(x1_2)
        x1_3 = self.conv1_3(x1_2)
        x1_4 = self.conv1_2(x1_2)
        x1_3_1 = x1_3.permute(0, 2, 3, 1).contiguous()
        x1_3_2 = x1_3_1.reshape((-1, height_Cloud_Band * width_Cloud_Band, channel_Cloud_Band))
        x1_4_1 = x1_4.permute(0, 2, 3, 1).contiguous()
        x1_4_2 = x1_4_1.reshape((-1, height_Cloud_Band * width_Cloud_Band, channel_Cloud_Band))
        x1_4_2T = torch.transpose(x1_4_2, dim0=2, dim1=1).contiguous()
        x1_3_2pro = torch.matmul(x1_3_2, x1_4_2T) / height_Cloud_Band
        x1_3_2pro = F.softmax(x1_3_2pro, dim=2)
        x1_3_2pro_max = torch.matmul(x1_3_2pro, x1_3_2)
        # x1_3_2pro_max = x1_3_2pro.mul(x1_3)
        x1_3_2RS = x1_3_2pro_max.reshape(-1, height_Cloud_Band, width_Cloud_Band, channel_Cloud_Band)
        x1_3_2RS = x1_3_2RS.permute(0, 3, 1, 2).contiguous()
        x1_eye = x1_3_2RS[:, :, 14:46, 14:46]                                                      #眼部相乘

        x1_eye = x1_eye.permute(0, 2, 3, 1).contiguous()
        x1_eye = x1_eye.reshape(-1, height_Eye * width_Eye, channel_Eye)
        x1_eyePro = torch.matmul(x_Eye_1, x1_eye)
        x1_eyePro = x1_eyePro.reshape(-1, height_Eye, width_Eye, channel_Eye)
        x1_eyePro = x1_eyePro.permute(0, 3, 1, 2).contiguous()
        x1_3_2RS[:, :, 14:46, 14:46] = x1_eyePro   #眼部填充
        x1_3_2RS[:, :, 14:46, 14:46] = x1_3_2RS[:, :, 14:46, 14:46] + x_Eye
        x_Eyes1 = x1_3_2RS[:, :, 14:46, 14:46]

        x1R = x1_3_2RS + x1_2
        x1R = F.interpolate(x1R, (480, 480), mode='bilinear', align_corners=True)
        # x_Eyes1 = F.interpolate(x_Eyes1, (256, 256), mode='bilinear', align_corners=True)
        x1R = x1R + x_4km
        x_Eyes1 = x1R[:, :, 112:368, 112:368]
        spat_edge1_1, spat_edge1_2 = self.spatial_edge_0_7(x1R)
        spat_edge2_1, spat_edge2_2 = self.spatial_edge_8_14(x1R)
        spec_edge1 = self.spectral_edge_0_7(x1R)
        spec_edge2 = self.spectral_edge_8_14(x1R)
        x_Cloud_Band_1_Import = F.interpolate(x_Cloud_Band_1_Import, (480, 480), mode='bilinear', align_corners=True)
        return x1R, x_Eyes1, spat_edge1_1, spat_edge1_2, spat_edge2_1, spat_edge2_2, spec_edge1, spec_edge2, x_Cloud_Band_1_Import




if __name__ == '__main__':
    # model = ResTFNet(14,7)
    lr = torch.randn(1, 14, 120, 120)
    hr = torch.randn(1, 7, 240, 240)
    # out = model(lr, hr)
    # print(out.shape)
