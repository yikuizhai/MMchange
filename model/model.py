import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Resnet50
from torch.nn import init
from einops.layers.torch import Rearrange






class Feature_extraction(nn.Module):
    def __init__(self, in_d=None, out_d=64):
        super(Feature_extraction, self).__init__()
        if in_d is None:
            in_d = [16, 24, 32, 96, 320]
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d
        # scale 2
        self.conv_scale2_c2 = nn.Sequential(
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale2_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 2, self.in_d[1], self.out_d)
        # scale 3
        self.conv_scale3_c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[1], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c3 = nn.Sequential(
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale3_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 3, self.in_d[2], self.out_d)
        # scale 4
        self.conv_scale4_c3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[2], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c4 = nn.Sequential(
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale4_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 3, self.in_d[3], self.out_d)
        # scale 5
        self.conv_scale5_c4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(self.in_d[3], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_scale5_c5 = nn.Sequential(
            nn.Conv2d(self.in_d[4], self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 2, self.in_d[4], self.out_d)

    def forward(self, c2, c3, c4, c5):
        # scale 2
        c2_s2 = self.conv_scale2_c2(c2)

        c3_s2 = self.conv_scale2_c3(c3)
        c3_s2 = F.interpolate(c3_s2, scale_factor=(2, 2), mode='bilinear')

        s2 = self.conv_aggregation_s2(torch.cat([c2_s2, c3_s2], dim=1), c2)
        # scale 3
        c2_s3 = self.conv_scale3_c2(c2)

        c3_s3 = self.conv_scale3_c3(c3)

        c4_s3 = self.conv_scale3_c4(c4)
        c4_s3 = F.interpolate(c4_s3, scale_factor=(2, 2), mode='bilinear')

        s3 = self.conv_aggregation_s3(torch.cat([c2_s3, c3_s3, c4_s3], dim=1), c3)
        # scale 4
        c3_s4 = self.conv_scale4_c3(c3)

        c4_s4 = self.conv_scale4_c4(c4)

        c5_s4 = self.conv_scale4_c5(c5)
        c5_s4 = F.interpolate(c5_s4, scale_factor=(2, 2), mode='bilinear')

        s4 = self.conv_aggregation_s4(torch.cat([c3_s4, c4_s4, c5_s4], dim=1), c4)
        # scale 5
        c4_s5 = self.conv_scale5_c4(c4)

        c5_s5 = self.conv_scale5_c5(c5)

        s5 = self.conv_aggregation_s5(torch.cat([c4_s5, c5_s5], dim=1), c5)

        return s2, s3, s4, s5

class FeatureFusionModule(nn.Module):
    def __init__(self, fuse_d, id_d, out_d):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.id_d = id_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_identity = nn.Conv2d(self.id_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

        self.enh = Feature_refinement(groups=8)
    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))
        c_out = self.enh(c_out)
        return c_out


class Attention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)


        return k1+k2


class Image_FeatureFusionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(Image_FeatureFusionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.relu = nn.ReLU(inplace=True)
        # branch 1
        self.conv_branch1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=7, dilation=7),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 2
        self.conv_branch2 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch2_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=5, dilation=5),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 3
        self.conv_branch3 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch3_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.BatchNorm2d(self.in_d)
        )
        # branch 4
        self.conv_branch4 = nn.Conv2d(self.in_d, self.in_d, kernel_size=1)
        self.conv_branch4_f = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.out_d)
        )
        self.conv_branch5 = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)
        self.att = Attention(self.in_d)
    def forward(self, x1, x2):
        # temporal fusion
        x = torch.abs(x1 - x2)
        x = self.att(x)
        # branch 1
        x_branch1 = self.conv_branch1(x)
        # branch 2
        x_branch2 = self.relu(self.conv_branch2(x) + x_branch1)
        x_branch2 = self.conv_branch2_f(x_branch2)
        # branch 3
        x_branch3 = self.relu(self.conv_branch3(x) + x_branch2)
        x_branch3 = self.conv_branch3_f(x_branch3)
        # branch 4
        x_branch4 = self.relu(self.conv_branch4(x) + x_branch3)
        x_branch4 = self.conv_branch4_f(x_branch4)
        x_out = self.relu(self.conv_branch5(x) + x_branch4)

        return x_out





class SupervisedAttentionModule(nn.Module):
    def __init__(self, mid_d):
        super(SupervisedAttentionModule, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(2, self.mid_d, kernel_size=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x.mul(context)
        x_out = self.conv2(x)

        return x_out, mask


class Decoder(nn.Module):
    def __init__(self, mid_d=320):
        super(Decoder, self).__init__()
        self.mid_d = mid_d
        # fusion
        self.sam_p5 = SupervisedAttentionModule(self.mid_d)
        self.sam_p4 = SupervisedAttentionModule(self.mid_d)
        self.sam_p3 = SupervisedAttentionModule(self.mid_d)
        self.conv_p4 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p3 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(self.mid_d, self.mid_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.mid_d),
            nn.ReLU(inplace=True)
        )
        self.cls = nn.Conv2d(self.mid_d, 1, kernel_size=1)

    def forward(self, d2, d3, d4, d5):
        # high-level
        p5, mask_p5 = self.sam_p5(d5)
        p4 = self.conv_p4(d4 + F.interpolate(p5, scale_factor=(2, 2), mode='bilinear'))

        p4, mask_p4 = self.sam_p4(p4)
        p3 = self.conv_p3(d3 + F.interpolate(p4, scale_factor=(2, 2), mode='bilinear'))

        p3, mask_p3 = self.sam_p3(p3)
        p2 = self.conv_p2(d2 + F.interpolate(p3, scale_factor=(2, 2), mode='bilinear'))
        mask_p2 = self.cls(p2)

        return p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5


class UpconvNet(nn.Module):
    def __init__(self):
        super(UpconvNet, self).__init__()
        self.upsample = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.out_d = 64
        self.conv = nn.Sequential(
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # Assuming x is of shape [32, 8, 64]
        # First, reshape to [32, 8, 1, 64] for upsampling
        x = x.unsqueeze(2)
        x = self.upsample(x)

        # Now x is of shape [32, 8, 64, 64]
        x = self.conv1(x)  # [32, 64, 64, 64]
        x = self.conv(x)

        return x

class DownsampleNet(nn.Module):
    def __init__(self):
        super(DownsampleNet, self).__init__()
        # Define a convolution layer with stride=2 to downsample the spatial dimensions
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.out_d = 64
        self.conv = nn.Sequential(
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        # x should be of shape [32, 64, 64, 64]
        x = self.conv1(x)  # Shape will be [32, 64, 32, 32]
        t3 = self.conv(x)
        x4 = self.conv2(t3)  # Shape will still be [32, 64, 32, 32]
        t4 = self.conv(x4)
        x5 = self.conv2(t4)  # Shape will still be [32, 64, 32, 32]
        t5 = self.conv(x5)
        return t3,t4,t5


class FeatureInteractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(FeatureInteractionModule, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh1 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_diff_enh2 = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.in_d * 2, self.in_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=1, bias=True),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # difference enhance
        x_sub = self.conv_sub(torch.abs(x1 - x2))
        x1 = self.conv_diff_enh1(x1.mul(x_sub) + x1)
        x2 = self.conv_diff_enh2(x2.mul(x_sub) + x2)
        # fusion
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv_cat(x_f)
        x = x_sub + x_f
        x = self.conv_dr(x)
        return x
class TDE(nn.Module):
    def __init__(self,out_dim = 64):
        super(TDE, self).__init__()
        self.dim = out_dim
        self.out_d = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.out_d, self.out_d, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )
        self.att = SimplifiedScaledDotProductAttention(d_model=self.dim, h=8)
        self.Up_conv1 = UpconvNet()
        self.FIM = FeatureInteractionModule(64, self.dim)
        self.Up_conv2 = DownsampleNet()
    def forward(self,textA,textB):
        B = textA.shape[0]
        textA1 = textA.reshape(B, -1 , self.dim)
        textB1 = textB.reshape(B, -1 , self.dim)
        text_diff = self.att(torch.abs(textA1-textB1),torch.abs(textA1-textB1),torch.abs(textA1-textB1))
        tA1 = self.Up_conv1(textA1)
        tB1 = self.Up_conv1(textB1)
        text_diff = self.Up_conv1(text_diff)
        text1 = self.FIM(tA1,tB1)

        t2 = self.conv(text1*text_diff)
        t3,t4,t5 = self.Up_conv2(t2)

        return t2,t3,t4,t5







class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class ITFF(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ITFF, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        initial = x + y
        cattn = self.ca(initial)
        sattn = self.sa(initial)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))
        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result

class Image_encoder(nn.Module):
    def __init__(self):
        super(Image_encoder,self).__init__()
        self.en_d = 32
        self.mid_d = self.en_d * 2
        channles = [16, 24, 32, 96, 320]
        self.backbone = Resnet50()
        self.fe = Feature_extraction(channles, self.mid_d)
    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.backbone(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.backbone(x2)
        x1_2, x1_3, x1_4, x1_5 = self.fe(x1_2, x1_3, x1_4, x1_5)
        x2_2, x2_3, x2_4, x2_5 = self.fe(x2_2, x2_3, x2_4, x2_5)
        return x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5

class Feature_refinement(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)

        return x
class IFR(nn.Module):
    def __init__(self, in_d=32, out_d=32):
        super(IFR,self).__init__()
        self.en_d = 32
        self.mid_d = self.en_d * 2
        self.in_d = in_d
        self.out_d = out_d
        # fusion
        self.ff2 = Image_FeatureFusionModule(self.in_d, self.out_d)
        self.ff3 = Image_FeatureFusionModule(self.in_d, self.out_d)
        self.ff4 = Image_FeatureFusionModule(self.in_d, self.out_d)
        self.ff5 = Image_FeatureFusionModule(self.in_d, self.out_d)
        self.enh = Feature_refinement(groups=8)
    def forward(self, x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5):
        c2 = self.ff2(x1_2, x2_2)
        c3 = self.ff3(x1_3, x2_3)
        c4 = self.ff4(x1_4, x2_4)
        c5 = self.ff5(x1_5, x2_5)
        c2_2 = self.enh(c2)
        c3_2 = self.enh(c3)
        c4_2 = self.enh(c4)
        c5_2 = self.enh(c5)
        return c2_2, c3_2, c4_2, c5_2
class BaseNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1):
        super(BaseNet, self).__init__()
        self.backbone = Resnet50()
        channles = [16, 24, 32, 96, 320]
        self.en_d = 32
        self.mid_d = self.en_d * 2

        self.decoder = Decoder(self.en_d * 2)
        self.Image_encoder = Image_encoder()
        self.IFR = IFR(self.mid_d, self.en_d * 2)
        self.TDE = TDE(self.en_d*2)
        self.ITFF = ITFF(self.en_d * 2)
    def forward(self, x1, x2, text_A, text_B):
        x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5 = self.Image_encoder(x1,x2)

        c2_2,c3_2,c4_2,c5_2 = self.IFR(x1_2, x1_3, x1_4, x1_5, x2_2, x2_3, x2_4, x2_5)

        t2,t3,t4,t5 = self.TDE(text_A,text_B)

        tc2 = self.ITFF(c2_2,t2)
        tc3 = self.ITFF(c3_2,t3)
        tc4 = self.ITFF(c4_2,t4)
        tc5 = self.ITFF(c5_2,t5)


        p2, p3, p4, p5, mask_p2, mask_p3, mask_p4, mask_p5 = self.decoder(tc2, tc3, tc4, tc5)

        # change map
        mask_p2 = F.interpolate(mask_p2, scale_factor=(4, 4), mode='bilinear')
        mask_p2 = torch.sigmoid(mask_p2)
        mask_p3 = F.interpolate(mask_p3, scale_factor=(8, 8), mode='bilinear')
        mask_p3 = torch.sigmoid(mask_p3)
        mask_p4 = F.interpolate(mask_p4, scale_factor=(16, 16), mode='bilinear')
        mask_p4 = torch.sigmoid(mask_p4)
        mask_p5 = F.interpolate(mask_p5, scale_factor=(32, 32), mode='bilinear')
        mask_p5 = torch.sigmoid(mask_p5)

        return mask_p2, mask_p3, mask_p4, mask_p5
