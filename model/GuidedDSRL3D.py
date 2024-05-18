# -*- coding: utf-8 -*-
import torch as pt

'''
class DoubleConv(pt.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),  
            pt.nn.LeakyReLU(inplace=True),
            pt.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),)

    def forward(self, input):
        return self.conv(input)

class SumLeakyReLU(pt.nn.Module):
    def __init__(self,channel):
        super(SumLeakyReLU, self).__init__()
        self.bnrelu = pt.nn.Sequential(
            pt.nn.LeakyReLU(inplace=True),)
    
    def forward(self,input):
        return self.bnrelu(input)

class ShortCut(pt.nn.Module):
    def __init__(self,in_ch,out_ch):
        super(ShortCut, self).__init__()
        self.shortcut = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch,out_ch,1),
            pt.nn.InstanceNorm3d(out_ch))
    
    def forward(self,input):
        return self.shortcut(input)


class DenseUNet(pt.nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(DenseUNet, self).__init__()
        
        self.conv1 = DoubleConv(in_channels, 64)
        self.shortcut1 = ShortCut(in_channels,64)
        self.sumrelu1 = SumLeakyReLU(64)
        self.pool1 = pt.nn.MaxPool3d((1,2,2))
        self.conv2 = DoubleConv(64, 128)
        self.shortcut2 = ShortCut(64,128)
        self.sumrelu2 = SumLeakyReLU(128)
        self.pool2 = pt.nn.MaxPool3d((1,2,2))
        self.conv3 = DoubleConv(128, 256)
        self.shortcut3 = ShortCut(128,256)
        self.sumrelu3 = SumLeakyReLU(256)
        self.pool3 = pt.nn.MaxPool3d((1,2,2))
        self.conv4 = DoubleConv(256, 512)
        self.shortcut4 = ShortCut(256,512)
        self.sumrelu4 = SumLeakyReLU(512)
        self.pool4 = pt.nn.MaxPool3d((1,2,2))
        self.conv5 = DoubleConv(512, 1024)
        self.shortcut5 = ShortCut(512,1024)
        self.sumrelu5 = SumLeakyReLU(1024)
        self.up6 = Deconv3D_Block(1024, 512, 4, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.shortcut6 = ShortCut(1024,512)
        self.sumrelu6 = SumLeakyReLU(512)
        self.up7 = Deconv3D_Block(512, 256, 4, stride=2)
        self.shortcut1 = ShortCut(512,256)
        self.conv7 = DoubleConv(512, 256)
        self.sumrelu7 = SumLeakyReLU(256)
        self.up8 = Deconv3D_Block(256, 128, 4, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.shortcut1 = ShortCut(256,128)
        self.sumrelu8 = SumLeakyReLU(128)
        self.up9 = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.shortcut1 = ShortCut(128,64)
        self.sumrelu9 = SumLeakyReLU(64)
        self.conv10 = pt.nn.Conv3d(64, out_channels, 1)

    def forward(self, x):
        c1 = self.sumrelu1(self.conv1(x)+self.shortcut1(x))
        p1 = self.pool1(c1)
        c2 = self.sumrelu2(self.conv2(p1)+self.shortcut2(p1))
        p2 = self.pool2(c2)
        c3 = self.sumrelu3(self.conv3(p2)+self.shortcut3(p2))
        p3 = self.pool3(c3)
        c4 = self.sumrelu4(self.conv4(p3)+self.shortcut4(p3))
        p4 = self.pool4(c4)
        c5 = self.sumrelu5(self.conv5(p4)+self.shortcut5(p4))
        up_6 = self.up6(c5)
        merge6 = pt.cat([up_6, c4], dim=1)
        c6 = self.sumrelu6(self.conv6(merge6)+self.shortcut6(merge6))
        up_7 = self.up7(c6)
        merge7 = pt.cat([up_7, c3], dim=1)
        c7 = self.sumrelu7(self.conv7(merge7)+self.shortcut7(merge7))
        up_8 = self.up8(c7)
        merge8 = pt.cat([up_8, c2], dim=1)
        c8 = self.sumrelu8(self.conv8(merge8)+self.shortcut8(merge8))
        up_9 = self.up9(c8)
        merge9 = pt.cat([up_9, c1], dim=1)
        c9 = self.sumrelu9(self.conv9(merge9)+self.shortcut9(merge9))
        c10 = self.conv10(c9)
        out = pt.nn.Sigmoid()(c10)
        return out

'''


class DoubleConv(pt.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),  
            pt.nn.LeakyReLU(inplace=True),
            pt.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),
            )

        self.residual_upsampler = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            pt.nn.InstanceNorm3d(out_ch))

        self.relu=pt.nn.LeakyReLU(inplace=True)

    def forward(self, input):
        return self.relu(self.conv(input)+self.residual_upsampler(input))

class EDSRConv(pt.nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EDSRConv, self).__init__()
        self.conv = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, 3, padding=1),
            pt.nn.InstanceNorm3d(out_ch),
            pt.nn.LeakyReLU(inplace=True),
            pt.nn.Conv3d(out_ch, out_ch, 3, padding=1),
            )

        self.residual_upsampler = pt.nn.Sequential(
            pt.nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False),
            pt.nn.InstanceNorm3d(out_ch)
            )

        # self.relu=pt.nn.LeakyReLU(inplace=True)

    def forward(self, input):
        return self.conv(input)+self.residual_upsampler(input)

class Deconv3D_Block(pt.nn.Module):
    
    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        
        super(Deconv3D_Block, self).__init__()
        
        self.deconv = pt.nn.Sequential(
                        pt.nn.ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel,kernel,kernel), 
                                    stride=(stride,stride,stride), padding=(padding, padding, padding), output_padding=0, bias=True),
                        pt.nn.LeakyReLU())
    
    def forward(self, x):
        
        return self.deconv(x)

class SubPixel_Block(pt.nn.Module):
    def __init__(self, upscale_factor=2):
        super(SubPixel_Block,self).__init__()

        self.subpixel=pt.nn.Sequential(
            PixelShuffle3d(upscale_factor),
            pt.nn.LeakyReLU()
        )

    def forward(self, x):
        return self.subpixel(x)

class PixelShuffle3d(pt.nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class MultiScaleConv(pt.nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv3x3=pt.nn.Conv3d(in_channels,out_channels,3,1,1)
        self.conv5x5=pt.nn.Conv3d(in_channels,out_channels//2,5,1,2)
        self.conv7x7=pt.nn.Conv3d(in_channels,out_channels//2,7,1,3)
        self.conv_2=pt.nn.Sequential(
            pt.nn.InstanceNorm3d(2*out_channels),
            pt.nn.LeakyReLU(inplace=True),
            pt.nn.Conv3d(2*out_channels,out_channels,3,1,1),
            pt.nn.InstanceNorm3d(out_channels),
        )
        self.residual_upsampler = pt.nn.Sequential(
            pt.nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
            pt.nn.InstanceNorm3d(out_channels))

        self.relu=pt.nn.LeakyReLU(inplace=True)
        
    def forward(self,x):
        conv3x3=self.conv3x3(x)
        conv5x5=self.conv5x5(x)
        conv7x7=self.conv7x7(x)
        hidden=pt.cat((conv3x3,conv5x5,conv7x7),dim=1)
        
        out=self.conv_2(hidden)
        return self.relu(out+self.residual_upsampler(x))


# class MultiScaleConv(pt.nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super().__init__()
#         self.conv1x1=pt.nn.Conv3d(in_channels,out_channels//4,1,1,0)
#         self.conv3x3=pt.nn.Conv3d(in_channels,out_channels//4,3,1,1)
#         self.conv5x5=pt.nn.Conv3d(in_channels,out_channels//4,5,1,2)
#         self.conv7x7=pt.nn.Conv3d(in_channels,out_channels//4,7,1,3)

#         self.conv1x1_2=pt.nn.Conv3d(out_channels,out_channels//4,1,1,0)
#         self.conv3x3_2=pt.nn.Conv3d(out_channels,out_channels//4,3,1,1)
#         self.conv5x5_2=pt.nn.Conv3d(out_channels,out_channels//4,5,1,2)
#         self.conv7x7_2=pt.nn.Conv3d(out_channels,out_channels//4,7,1,3)

#     def forward(self,x):
#         conv1x1=self.conv1x1(x)
#         conv3x3=self.conv3x3(x)
#         conv5x5=self.conv5x5(x)
#         conv7x7=self.conv7x7(x)
#         hidden=pt.cat((conv1x1,conv3x3,conv5x5,conv7x7),dim=1)
        
#         conv1x1=self.conv1x1_2(hidden)
#         conv3x3=self.conv3x3_2(hidden)
#         conv5x5=self.conv5x5_2(hidden)
#         conv7x7=self.conv7x7_2(hidden)

#         out=pt.cat((conv1x1,conv3x3,conv5x5,conv7x7),dim=1)
#         return out


class GuidedDSRL3D(pt.nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(GuidedDSRL3D, self).__init__()
        self.conv1 = DoubleConv(in_channels, 32)
        self.pool1 = pt.nn.MaxPool3d((2,2,2))
        self.conv2 = DoubleConv(32, 32)
        self.pool2 = pt.nn.MaxPool3d((2,2,2))
        self.conv3 = DoubleConv(32, 64)
        self.pool3 = pt.nn.MaxPool3d((2,2,2))
        self.conv4 = DoubleConv(64, 128)
        self.pool4 = pt.nn.MaxPool3d((2,2,2))
        self.conv5 = DoubleConv(128, 256)
        self.up6_seg = Deconv3D_Block(256+64, 128, 4, stride=2)
        self.conv6_seg = DoubleConv(256, 128)
        self.up7_seg = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_seg = DoubleConv(128, 64)
        self.up8_seg = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_seg = DoubleConv(64, 32)
        self.up9_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_seg = DoubleConv(64, 32)
        self.up10_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_seg = DoubleConv(32, 16)
        self.conv11_seg = pt.nn.Conv3d(16, out_channels, 1)

        self.up6_sr = Deconv3D_Block(256+64, 128, 4, stride=2)
        self.conv6_sr = DoubleConv(256, 128)
        self.up7_sr = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_sr = DoubleConv(128, 64)
        self.up8_sr = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_sr = DoubleConv(64, 32)
        self.up9_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_sr = DoubleConv(64, 32)
        self.up10_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_sr = DoubleConv(32, 16)
        self.conv11_sr = pt.nn.Conv3d(16, out_channels, 1)

        self.high_freq_extract=pt.nn.Sequential(
            DoubleConv(in_channels,16),
            pt.nn.MaxPool3d((2,2,2)),
            DoubleConv(16,32),
            pt.nn.MaxPool3d((2,2,2)),
            DoubleConv(32,64),
            pt.nn.MaxPool3d((2,2,2)),
            DoubleConv(64,64),
        )

    def forward(self, x, guide):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        hfe_seg=self.high_freq_extract(guide)

        up_6_seg = self.up6_seg(pt.cat([c5,hfe_seg], dim=1))
        merge6_seg = pt.cat([up_6_seg, c4], dim=1)
        c6_seg = self.conv6_seg(merge6_seg)
        up_7_seg = self.up7_seg(c6_seg)
        merge7_seg = pt.cat([up_7_seg, c3], dim=1)
        c7_seg = self.conv7_seg(merge7_seg)
        up_8_seg = self.up8_seg(c7_seg)
        merge8_seg = pt.cat([up_8_seg, c2], dim=1)
        c8_seg = self.conv8_seg(merge8_seg)
        up_9_seg = self.up9_seg(c8_seg)
        merge9_seg = pt.cat([up_9_seg, c1], dim=1)
        c9_seg = self.conv9_seg(merge9_seg)
        up_10_seg = self.up10_seg(c9_seg)
        c10_seg = self.conv10_seg(up_10_seg)
        # c11_seg = self.pointwise(c10_seg)
        c11_seg = self.conv11_seg(c10_seg)
        out_seg = pt.nn.Sigmoid()(c11_seg)

        hfe_sr=self.high_freq_extract(guide)

        up_6_sr = self.up6_sr(pt.cat([c5,hfe_sr], dim=1))
        merge6_sr = pt.cat([up_6_sr, c4], dim=1)
        c6_sr = self.conv6_sr(merge6_sr)
        up_7_sr = self.up7_sr(c6_sr)
        merge7_sr = pt.cat([up_7_sr, c3], dim=1)
        c7_sr = self.conv7_sr(merge7_sr)
        up_8_sr = self.up8_sr(c7_sr)
        merge8_sr = pt.cat([up_8_sr, c2], dim=1)
        c8_sr = self.conv8_sr(merge8_sr)
        up_9_sr = self.up9_sr(c8_sr)
        merge9_sr = pt.cat([up_9_sr, c1], dim=1)
        c9_sr = self.conv9_sr(merge9_sr)
        up_10_sr = self.up10_sr(c9_sr)
        c10_sr = self.conv10_sr(up_10_sr)
        c11_sr = self.conv11_sr(c10_sr)
        out_sr = pt.nn.ReLU()(c11_sr)

        return out_seg, out_sr


class  GuidedPFSegFull(pt.nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(GuidedPFSegFull, self).__init__()
        self.conv1 = MultiScaleConv(in_channels, 32)
        self.pool1 = pt.nn.MaxPool3d((2,2,2))
        self.conv2 = MultiScaleConv(32, 32)
        self.pool2 = pt.nn.MaxPool3d((2,2,2))
        self.conv3 = MultiScaleConv(32, 64)
        self.pool3 = pt.nn.MaxPool3d((2,2,2))
        self.conv4 = MultiScaleConv(64, 128)
        self.pool4 = pt.nn.MaxPool3d((2,2,2))
        self.conv5 = MultiScaleConv(128, 256)
        self.up6_seg = Deconv3D_Block(256+64, 128, 4, stride=2)
        self.conv6_seg = DoubleConv(256, 128)
        self.up7_seg = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_seg = DoubleConv(128, 64)
        self.up8_seg = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_seg = DoubleConv(64, 32)
        self.up9_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_seg = DoubleConv(64, 32)
        self.up10_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_seg = DoubleConv(32, 16)
        self.conv11_seg = pt.nn.Conv3d(16, out_channels, 1)

        self.up6_sr = Deconv3D_Block(256+64, 128, 4, stride=2)
        self.conv6_sr = DoubleConv(256, 128)
        self.up7_sr = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_sr = DoubleConv(128, 64)
        self.up8_sr = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_sr = DoubleConv(64, 32)
        self.up9_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_sr = DoubleConv(64, 32)
        self.up10_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_sr = DoubleConv(32, 16)
        self.conv11_sr = pt.nn.Conv3d(16, out_channels, 1)

        self.ssgm=pt.nn.Sequential(
            MultiScaleConv(in_channels,16),
            pt.nn.MaxPool3d((2,2,2)),
            MultiScaleConv(16,32),
            pt.nn.MaxPool3d((2,2,2)),
            MultiScaleConv(32,64),
            pt.nn.MaxPool3d((2,2,2)),
            MultiScaleConv(64,64),
        )

    def forward(self, x, guide):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        hfe_seg=self.ssgm(guide)

        up_6_seg = self.up6_seg(pt.cat([c5,hfe_seg], dim=1))
        merge6_seg = pt.cat([up_6_seg, c4], dim=1)
        c6_seg = self.conv6_seg(merge6_seg)
        up_7_seg = self.up7_seg(c6_seg)
        merge7_seg = pt.cat([up_7_seg, c3], dim=1)
        c7_seg = self.conv7_seg(merge7_seg)
        up_8_seg = self.up8_seg(c7_seg)
        merge8_seg = pt.cat([up_8_seg, c2], dim=1)
        c8_seg = self.conv8_seg(merge8_seg)
        up_9_seg = self.up9_seg(c8_seg)
        merge9_seg = pt.cat([up_9_seg, c1], dim=1)
        c9_seg = self.conv9_seg(merge9_seg)
        up_10_seg = self.up10_seg(c9_seg)
        c10_seg = self.conv10_seg(up_10_seg)
        # c11_seg = self.pointwise(c10_seg)
        c11_seg = self.conv11_seg(c10_seg)
        out_seg = pt.nn.Sigmoid()(c11_seg)

        hfe_sr=self.ssgm(guide) # This line can be removed, as hfe_sr=hfe_seg

        up_6_sr = self.up6_sr(pt.cat([c5,hfe_sr], dim=1))
        merge6_sr = pt.cat([up_6_sr, c4], dim=1)
        c6_sr = self.conv6_sr(merge6_sr)
        up_7_sr = self.up7_sr(c6_sr)
        merge7_sr = pt.cat([up_7_sr, c3], dim=1)
        c7_sr = self.conv7_sr(merge7_sr)
        up_8_sr = self.up8_sr(c7_sr)
        merge8_sr = pt.cat([up_8_sr, c2], dim=1)
        c8_sr = self.conv8_sr(merge8_sr)
        up_9_sr = self.up9_sr(c8_sr)
        merge9_sr = pt.cat([up_9_sr, c1], dim=1)
        c9_sr = self.conv9_sr(merge9_sr)
        up_10_sr = self.up10_sr(c9_sr)
        c10_sr = self.conv10_sr(up_10_sr)
        c11_sr = self.conv11_sr(c10_sr)
        out_sr = pt.nn.ReLU()(c11_sr)

        return out_seg, out_sr

class  GuidedMultiScaleDSRL3D_noguide(pt.nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(GuidedMultiScaleDSRL3D_noguide, self).__init__()
        self.conv1 = MultiScaleConv(in_channels, 32)
        self.pool1 = pt.nn.MaxPool3d((2,2,2))
        self.conv2 = MultiScaleConv(32, 32)
        self.pool2 = pt.nn.MaxPool3d((2,2,2))
        self.conv3 = MultiScaleConv(32, 64)
        self.pool3 = pt.nn.MaxPool3d((2,2,2))
        self.conv4 = MultiScaleConv(64, 128)
        self.pool4 = pt.nn.MaxPool3d((2,2,2))
        self.conv5 = MultiScaleConv(128, 256)
        self.up6_seg = Deconv3D_Block(256, 128, 4, stride=2)
        self.conv6_seg = DoubleConv(256, 128)
        self.up7_seg = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_seg = DoubleConv(128, 64)
        self.up8_seg = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_seg = DoubleConv(64, 32)
        self.up9_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_seg = DoubleConv(64, 32)
        self.up10_seg = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_seg = pt.nn.Sequential(
            DoubleConv(32, 16),
            pt.nn.Upsample([128,192,192],mode='trilinear',align_corners=True)
        )
        self.conv11_seg = pt.nn.Conv3d(16, out_channels, 1)

        self.up6_sr = Deconv3D_Block(256, 128, 4, stride=2)
        self.conv6_sr = DoubleConv(256, 128)
        self.up7_sr = Deconv3D_Block(128, 64, 4, stride=2)
        self.conv7_sr = DoubleConv(128, 64)
        self.up8_sr = Deconv3D_Block(64, 32, 4, stride=2)
        self.conv8_sr = DoubleConv(64, 32)
        self.up9_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv9_sr = DoubleConv(64, 32)
        self.up10_sr = Deconv3D_Block(32, 32, 4, stride=2)
        self.conv10_sr = pt.nn.Sequential(
            DoubleConv(32, 16),
            pt.nn.Upsample([128,192,192],mode='trilinear',align_corners=True)
        )
        self.conv11_sr = pt.nn.Conv3d(16, out_channels, 1)

        # self.ssgm=pt.nn.Sequential(
        #     MultiScaleConv(in_channels,16),
        #     # pt.nn.MaxPool3d((2,2,2)),
        #     MultiScaleConv(16,32),
        #     # pt.nn.MaxPool3d((2,2,2)),
        #     MultiScaleConv(32,64),
        #     pt.nn.MaxPool3d((2,2,2)),
        #     MultiScaleConv(64,64),
        # )

    def forward(self, x, guide):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)

        # hfe_seg=self.ssgm(guide)

        up_6_seg = self.up6_seg(c5)
        merge6_seg = pt.cat([up_6_seg, c4], dim=1)
        c6_seg = self.conv6_seg(merge6_seg)
        up_7_seg = self.up7_seg(c6_seg)
        merge7_seg = pt.cat([up_7_seg, c3], dim=1)
        c7_seg = self.conv7_seg(merge7_seg)
        up_8_seg = self.up8_seg(c7_seg)
        merge8_seg = pt.cat([up_8_seg, c2], dim=1)
        c8_seg = self.conv8_seg(merge8_seg)
        up_9_seg = self.up9_seg(c8_seg)
        merge9_seg = pt.cat([up_9_seg, c1], dim=1)
        c9_seg = self.conv9_seg(merge9_seg)
        up_10_seg = self.up10_seg(c9_seg)
        c10_seg = self.conv10_seg(up_10_seg)
        # c11_seg = self.pointwise(c10_seg)
        c11_seg = self.conv11_seg(c10_seg)
        out_seg = pt.nn.Sigmoid()(c11_seg)

        # hfe_sr=self.ssgm(guide)

        up_6_sr = self.up6_sr(c5)
        merge6_sr = pt.cat([up_6_sr, c4], dim=1)
        c6_sr = self.conv6_sr(merge6_sr)
        up_7_sr = self.up7_sr(c6_sr)
        merge7_sr = pt.cat([up_7_sr, c3], dim=1)
        c7_sr = self.conv7_sr(merge7_sr)
        up_8_sr = self.up8_sr(c7_sr)
        merge8_sr = pt.cat([up_8_sr, c2], dim=1)
        c8_sr = self.conv8_sr(merge8_sr)
        up_9_sr = self.up9_sr(c8_sr)
        merge9_sr = pt.cat([up_9_sr, c1], dim=1)
        c9_sr = self.conv9_sr(merge9_sr)
        up_10_sr = self.up10_sr(c9_sr)
        c10_sr = self.conv10_sr(up_10_sr)
        c11_sr = self.conv11_sr(c10_sr)
        out_sr = pt.nn.ReLU()(c11_sr)

        return out_seg, out_sr