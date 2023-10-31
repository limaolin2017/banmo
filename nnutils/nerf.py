# ReadMe
'''
NeRF是干什么用的？
NeRF用于合成3D场景的新视图。它将空间位置和视角方向编码为颜色和密度，
然后通过体积渲染来合成新的图像。
它可以用于虚拟现实、增强现实、视觉特效以及任何需要精确和真实感3D渲染的应用。


NeRF是如何建模的？
NeRF使用多层感知机（MLP）来建模一个场景。
它将3D坐标和观察方向作为输入，并输出对应的颜色（RGB）值和体积密度（sigma）。
这个文件中的 NeRF 类定义了这个基本的结构，包括对空间位置（xyz）和方向（dir）的编码层，以及最终的输出层。

NeRF通过神经网络模型来学习场景的连续体积表示。它采用以下几个步骤建模：

位置编码（Positional Encoding）: 将输入的连续坐标（如空间位置和视线方向）通过三角函数映射到一个高维空间，以捕获更细微的变化。
网络架构（Network Architecture）: 使用多层感知机（MLP），该模型预测通过3D空间中每个点的颜色和密度（即场景的体积表示）。
渲染（Rendering）: 使用体积渲染技术，通过在每条光线上积分来合成图像。

nerf.py 文件包含了用于实现神经辐射场（NeRF）模型及其变种的类和函数。NeRF 是一种用于3D场景重建和视图合成的深度学习方法，
通过对场景的密集采样和神经网络的训练，可以从一组稀疏的视角生成新的视角图像。

NeRF的输入输出是什么？
输入：

空间位置（XYZ坐标）: 表示场景中的点。
视线方向: 表示观察该点的方向。
输出：

颜色（RGB值）: 预测的每个点的颜色。
密度（Sigma值）: 预测的每个点的体积密度。

'''




# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import pdb
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from pytorch3d import transforms
import trimesh
from nnutils.geom_utils import fid_reindex

#Embedding: 进行位置编码。
class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True, alpha=None):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.nfuncs = len(self.funcs)
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)
        if alpha is None:
            self.alpha = self.N_freqs
        else: self.alpha = alpha

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        # consine features
        if self.N_freqs>0:
            shape = x.shape
            bs = shape[0]
            input_dim = shape[-1]
            output_dim = input_dim*(1+self.N_freqs*self.nfuncs)
            out_shape = shape[:-1] + ((output_dim),)
            device = x.device

            x = x.view(-1,input_dim)
            out = []
            for freq in self.freq_bands:
                for func in self.funcs:
                    out += [func(freq*x)]
            out =  torch.cat(out, -1)

            ## Apply the window w = 0.5*( 1+cos(pi + pi clip(alpha-j)) )
            out = out.view(-1, self.N_freqs, self.nfuncs, input_dim)
            window = self.alpha - torch.arange(self.N_freqs).to(device)
            window = torch.clamp(window, 0.0, 1.0)
            window = 0.5 * (1 + torch.cos(np.pi * window + np.pi))
            window = window.view(1,-1, 1, 1)
            out = window * out
            out = out.view(-1,self.N_freqs*self.nfuncs*input_dim)

            out = torch.cat([x, out],-1)
            out = out.view(out_shape)
        else: out = x
        return out


# NeRF: 基础的NeRF网络结构。
class NeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27,
                 out_channels=3, 
                 skips=[4], raw_feat=False, init_beta=1./100, 
                 activation=nn.ReLU(True), in_channels_code=0):
        """
        adapted from https://github.com/kwea123/nerf_pl/blob/master/models/nerf.py
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        in_channels_code: only used for nerf_skin,
        """
        super(NeRF, self).__init__()

        # D 网络的层数，用于密度（sigma）编码器。
        '''
        "D"指的是网络中用于密度（或称为sigma）编码的层数。它是一个参数，
        表示在构建用于估计场景中每个点的密度（这对于渲染过程来说是至关重要的）的网络时应使用多少层。
        代码中的 D=8 表示使用了8层网络来进行密度编码。每一层都会接收输入特征，并输出到下一层，直到最后一层输出密度值（sigma）。这些层可以帮助网络学习场景的复杂几何特性。
        '''
        self.D = D

        # 每一层的隐藏单元数。
        '''
        在神经网络中，"隐藏单元"（通常也称为"神经元"或"节点"）指的是网络中非输入层的单个处理元素。每个隐藏单元接收来自前一层的输入，并将其与一组权重相乘（加上一个偏置），
        然后通过一个激活函数来产生输出，该输出将传递到下一层。
        这段代码中提到的"每一层的隐藏单元数"指的是每个隐藏层中的神经元数量。
        在NeRF类的构造函数中，参数W表示每个隐藏层的宽度，即每一层中隐藏单元的数量。在这个实现中，所有隐藏层都被设置为具有相同数量的隐藏单元W。
        例如，如果W设置为256，这意味着每个隐藏层都有256个神经元。每个隐藏层的输入将与一个W×W的权重矩阵相乘（对于跳过连接的层，权重矩阵的大小将为
        (W+输入特征数)×W），这样每个层都能输出一个宽度为W的向量。
        这些隐藏单元是构成深度学习模型中隐藏层的基本构建块，允许模型学习和表示数据中的复杂模式和关系。
        '''
        self.W = W

        # in_channels_xyz和in_channels_dir分别代表了空间位置和方向输入的特征数量。
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.in_channels_code = in_channels_code
        # skips是一个列表，指示应在哪些层添加跳跃连接，以提高模型性能。
        self.skips = skips
        self.use_xyz = False

        # xyz encoding layers
        self.weights_reg = []
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i+1}")
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
                self.weights_reg.append(f"xyz_encoding_{i+1}")
            else:
                layer = nn.Linear(W, W)
        
            # activation是激活函数，这里默认使用ReLU。
            layer = nn.Sequential(layer, activation)
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(W, W)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(W+in_channels_dir, W//2),
                                activation)

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, out_channels),
                        )
        # raw_feat是一个标志，用于指示模型是否应该输出原始特征（未经激活函数处理的）。
        self.raw_feat = raw_feat
        
        # init_beta是初始化密度（sigma）预测的权重的参数。
        self.beta = torch.Tensor([init_beta]) # logbeta
        self.beta = nn.Parameter(self.beta)
        
#        for m in self.modules():
#            if isinstance(m, nn.Linear):
#                if hasattr(m.weight,'data'):
#                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x ,xyz=None, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)
            raw_feat: does not apply sigmoid

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, 0], dim=-1)

        xyz_ = input_xyz
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)

        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir], -1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        rgb = self.rgb(dir_encoding)
        if self.raw_feat:
            out = rgb
        else:
            rgb = rgb.sigmoid()
            out = torch.cat([rgb, sigma], -1)
        return out
# Transhead, SE3head, RTHead, 
# FrameCode, RTExplicit, RTExpMLP, ScoreHead, NeRFUnc: 为不同的场景或特殊用途定制的NeRF变种。
class Transhead(NeRF):
    """
    translation head
    """
    def __init__(self, **kwargs):
        super(Transhead, self).__init__(**kwargs)

    def forward(self, x, xyz=None,sigma_only=False):
        flow = super(Transhead, self).forward(x, sigma_only=sigma_only)
        flow = flow*0.1
        return flow

class SE3head(NeRF):
    """
    modify the output to be rigid transforms per point
    modified from Nerfies
    """
    def __init__(self, **kwargs):
        super(SE3head, self).__init__(**kwargs)
        self.use_xyz=True

    def forward(self, x, xyz=None,sigma_only=False):
        x = super(SE3head, self).forward(x, sigma_only=sigma_only)
        x = x.view(-1,9)
        rotation, pivot, translation = x.split([3,3,3],-1)
        pivot = pivot*0.1
        translation = translation*0.1
        
        shape = xyz.shape
        warped_points = xyz.view(-1,3).clone()
        warped_points = warped_points + pivot
        rotmat = transforms.so3_exponential_map(rotation)
        warped_points = rotmat.matmul(warped_points[...,None])[...,0]
        warped_points = warped_points - pivot
        warped_points = warped_points + translation

        flow = warped_points.view(shape) - xyz
        return flow

class RTHead(NeRF):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, use_quat, **kwargs):
        super(RTHead, self).__init__(**kwargs)
        # use quaternion when estimating full rotation
        # use exponential map when estimating delta rotation
        self.use_quat=use_quat
        if self.use_quat: self.num_output=7
        else: self.num_output=6

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(RTHead, self).forward(x)
        bs = x.shape[0]
        rts = x.view(-1,self.num_output)  # bs B,x
        B = rts.shape[0]//bs

        tmat= rts[:,0:3] *0.1

        if self.use_quat:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        else:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts
            

class FrameCode(nn.Module):
    """
    frame index and video index to code
    """
    def __init__(self, num_freq, embedding_dim, vid_offset, scale=1):
        super(FrameCode, self).__init__()
        self.vid_offset = vid_offset
        self.num_vids = len(vid_offset)-1
        # compute maximum frequency:64-127 frame=>10
        max_ts = (self.vid_offset[1:] - self.vid_offset[:-1]).max()
        self.num_freq = 2*int(np.log2(max_ts))-2
#        self.num_freq = num_freq

        self.fourier_embed = Embedding(1,num_freq,alpha=num_freq)
        self.basis_mlp = nn.Linear(self.num_vids*self.fourier_embed.out_channels,
                                embedding_dim)
        self.scale = scale # input scale factor

    def forward(self, fid):
        """
        fid->code: N->N,embedding_dim
        """
        bs = fid.shape[0]
        vid, tid = fid_reindex(fid, self.num_vids, self.vid_offset)
        tid = tid*self.scale
        tid = tid.view(bs,1)
        vid = vid.view(bs,1)
        coeff = self.fourier_embed(tid) # N, n_channels
        vid = F.one_hot(vid, num_classes=self.num_vids) # N, 1, num_vids
        # pad zeros for each
        coeff = coeff[...,None] * vid # N, n_channels, num_vids
        coeff = coeff.view(bs, -1)
        code = self.basis_mlp(coeff)
        return code

class RTExplicit(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, delta=False, rand=True):
        super(RTExplicit, self).__init__()
        self.max_t = max_t
        self.delta = delta

        # initialize rotation
        trans = torch.zeros(max_t, 3)
        if delta:
            rot = torch.zeros(max_t, 3) 
        else:
            if rand:
                rot = torch.rand(max_t, 4) * 2 - 1
            else:
                rot = torch.zeros(max_t, 4)
                rot[:,0] = 1
        se3 = torch.cat([trans, rot],-1)

        self.se3 = nn.Parameter(se3)
        self.num_output = se3.shape[-1]


    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        bs = x.shape[0]
        x = self.se3[x] # bs B,x
        rts = x.view(-1,self.num_output)
        B = rts.shape[0]//bs
        
        tmat= rts[:,0:3] *0.1

        if self.delta:
            rot=rts[:,3:6]
            rmat = transforms.so3_exponential_map(rot)
        else:
            rquat=rts[:,3:7]
            rquat=F.normalize(rquat,2,-1)
            rmat=transforms.quaternion_to_matrix(rquat) 
        rmat = rmat.view(-1,9)

        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(bs,1,-1)
        return rts

class RTExpMLP(nn.Module):
    """
    index rigid transforms from a dictionary
    """
    def __init__(self, max_t, num_freqs, t_embed_dim, data_offset, delta=False):
        super(RTExpMLP, self).__init__()
        #self.root_code = nn.Embedding(max_t, t_embed_dim)
        self.root_code = FrameCode(num_freqs, t_embed_dim, data_offset, scale=0.1)

        self.base_rt = RTExplicit(max_t, delta=delta,rand=False)
        #self.base_rt = RTHead(use_quat=True, 
        #            D=2, W=64,
        #            in_channels_xyz=t_embed_dim,in_channels_dir=0,
        #            out_channels=7, raw_feat=True)
        #self.base_rt = nn.Sequential(self.root_code, self.base_rt)
        self.mlp_rt = RTHead(use_quat=False, 
                    in_channels_xyz=t_embed_dim,in_channels_dir=0,
                    out_channels=6, raw_feat=True)
        self.delta_rt = nn.Sequential(self.root_code, self.mlp_rt)


    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        base_rts = self.base_rt(x)
        delt_rts = self.delta_rt(x)

        # magnify gradient by 10x
        base_rts = base_rts * 10 - (base_rts*9).detach()
        
        rmat = base_rts[:,0,:9].view(-1,3,3)
        tmat = base_rts[:,0,9:12]
        
        delt_rmat = delt_rts[:,0,:9].view(-1,3,3)
        delt_tmat = delt_rts[:,0,9:12]
    
        tmat = tmat + rmat.matmul(delt_tmat[...,None])[...,0]
        rmat = rmat.matmul(delt_rmat)
        
        rmat = rmat.view(-1,9)
        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(-1,1,12)
        return rts

class ScoreHead(NeRF):
    """
    modify the output to be rigid transforms
    """
    def __init__(self, recursion_level, **kwargs):
        super(ScoreHead, self).__init__(**kwargs)
        grid= generate_healpix_grid(recursion_level=recursion_level)
        self.register_buffer('grid', grid)
        self.num_scores = self.grid.shape[0]

    def forward(self, x):
        # output: NxBx(9 rotation + 3 translation)
        x = super(ScoreHead, self).forward(x)
        bs = x.shape[0]
        x = x.view(-1,self.num_scores+3)  # bs B,x

        # do not use tmat since it is not trained
        tmat = x[:,0:3]*0.
        scores=x[:,3:]
        if self.training:
            return scores, self.grid
        else:
            scores = scores.view(bs,-1,1)
            rmat = self.grid[None].repeat(bs,1,1,1)
            tmat = tmat[:,None].repeat(1,self.num_scores,1)
            rmat = rmat.view(bs,-1,9)
            rts = torch.cat([scores,rmat, tmat],-1)
            rts = rts.view(bs,self.num_scores,-1)
            return rts

class NeRFUnc(NeRF):
    """
    nerf uncertainty
    """
    def __init__(self, **kwargs):
        super(NeRFUnc, self).__init__(**kwargs)

    def forward(self, x, xyz=None,sigma_only=False):
        unc = super(NeRFUnc, self).forward(x, sigma_only=sigma_only)
        return unc

# ResNetConv 和 Encoder: 用于图像特征提取的卷积神经网络结构。
class ResNetConv(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    def __init__(self, in_channels):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        if in_channels!=3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), 
                                    stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc=None

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

class Encoder(nn.Module):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    Current:
    Resnet with 4 blocks (x32 spatial dim reduction)
    Another conv with stride 2 (x64)
    This is sent to 2 fc layers with final output nz_feat.
    """

    def __init__(self, input_shape, in_channels=3,out_channels=128, batch_norm=True):
        super(Encoder, self).__init__()
        self.resnet_conv = ResNetConv(in_channels=in_channels)
        self.conv1 = conv2d(batch_norm, 512, 128, stride=1, kernel_size=3)
        #net_init(self.conv1)

    def forward(self, img):
        feat = self.resnet_conv.forward(img) # 512,4,4
        feat = self.conv1(feat) # 128,4,4
        feat = F.max_pool2d(feat, 4, 4)
        feat = feat.view(img.size(0), -1)
        return feat

## 2D convolution layers
def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    """
    adapted from https://github.com/shubhtuls/factored3d/blob/master/nnutils/net_blocks.py
    """
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )

def grab_xyz_weights(nerf_model, clone=False):
    """
    zero grad for coarse component connected to inputs, 
    and return intermediate params
    """
    param_list = []
    input_layers=[0]+nerf_model.skips

    input_wt_names = []
    for layer in input_layers:
        input_wt_names.append(f"xyz_encoding_{layer+1}.0.weight")

    for name,p in nerf_model.named_parameters():
        if name in input_wt_names:
            # equiv since the wt after pos_dim does not change
            if clone:
                param_list.append(p.detach().clone()) 
            else:
                param_list.append(p) 
            ## get the weights according to coarse posec
            ## 63 = 3 + 60
            ## 60 = (num_freqs, 2, 3)
            #out_dim = p.shape[0]
            #pos_dim = nerf_model.in_channels_xyz-nerf_model.in_channels_code
            #param_list.append(p[:,:pos_dim]) # 
    return param_list

