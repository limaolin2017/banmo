# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# adopted from nerf-pl
import numpy as np
import pdb
import torch
import torch.nn.functional as F
from pytorch3d import transforms

from nnutils.geom_utils import lbs, Kmatinv, mat2K, pinhole_cam, obj_to_cam,\
                               vec_to_sim3, rtmat_invert, rot_angle, mlp_skinning,\
                               bone_transform, skinning, vrender_flo, \
                               gauss_mlp_skinning, diff_flo
from nnutils.loss_utils import elastic_loss, visibility_loss, feat_match_loss,\
                                kp_reproj_loss, compute_pts_exp, kp_reproj, evaluate_mlp

def render_rays(models,         # models 是 NeRF 模型的列表，通常包括一个粗模型和一个精细模型。
                embeddings,     # embeddings 是位置和方向的嵌入模型列表。
                rays,           # rays 包含了光线的原点、方向以及近、远深度界限。
                N_samples=64,   # N_samples 是每条光线上采样点的数量。
                use_disp=False, # use_disp 指示是否在视差空间（即逆深度）中采样。
                perturb=0,      # 其他参数如 perturb、noise_std、chunk 等用于控制渲染细节。
                noise_std=1,
                chunk=1024*32,
                obj_bound=None,
                use_fine=False,
                img_size=None,
                progress=None,
                opts=None,
                render_vis=False,
                ):
    """
    Render rays by computing the output of @model applied on @rays

    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins, directions and near, far depth bounds
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        chunk: the chunk size in batched inference

    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """
    # 样本数修改: 如果使用精细模型，将采样数减半，以便于后面进行重要性采样。
    '''
    函数的目的：
    在这个函数中，rays 参数包含了光线的原点 rays_o、方向 rays_d，以及每条光线的近处和远处深度界限 near 和 far。这些值用于沿光线计算样本点的位置，这些样本点之后会被送入神经网络模型中以预测其颜色和透明度（密度）。

    此函数的核心是对输入的光线执行一个过程，这个过程包括：

    将光线的起点和方向进行编码。
    在每条光线上均匀采样多个点。
    根据采样点的位置和方向，使用神经网络模型计算颜色和密度。
    可选地，使用一个细化的网络对结果进行改进



    为什么如果使用精细模型，需要对采样数进行减半？

    在渲染过程中，NeRF通常有两个网络：一个粗略（coarse）网络和一个精细（fine）网络。
    粗略网络用来快速估计场景的整体结构，而精细网络则用来增强细节。
    这里的策略是首先使用粗略网络对光线路径上的点进行采样，然后基于粗略网络的预测，
    使用重要性采样（importance sampling）技术选择更可能贡献更多信息的点进行精细网络的采样。

    重要性采样是一个统计技术，在这个上下文中，它利用了粗略网络输出的权重来指导在哪些区域进行更密集的采样。
    这意味着在粗略网络已经选出了一些有用的点之后，精细网络不需要同样数量的采样点，
    因为它只需要对这些选出的点进行详细处理。因此，原始的采样点数会减半，为重要性采样腾出空间，
    最后再将重要性采样得到的点和原始采样点合并，以进行精细网络的预测。这种方法可以提高效率，
    因为它避免了在可能不那么重要的区域上浪费计算资源。

    是否一定要减半:
    并不一定要减半。重要性采样的数量是一个可以调整的超参数，取决于你想要在计算效率和渲染质量之间做出的权衡。
    减半是一个常用的选择，因为它在保持模型性能的同时减少了计算量。你完全可以尝试减少三分之一或其他比例，
    然后根据实际情况来调整这个参数，以找到最适合你项目需求的设置。
    
    '''
    if use_fine: N_samples = N_samples//2 # use half samples to importance sample

    # 提取模型和嵌入: 从传入的列表中提取出 xyz 和方向的嵌入模型。
    # Extract models from lists
    embedding_xyz = embeddings['xyz']
    embedding_dir = embeddings['dir']

    # 分解输入: 从 rays 字典中分解出原点 rays_o 和方向 rays_d，以及近、远界限 near 和 far。
    # Decompose the inputs
    rays_o = rays['rays_o']
    rays_d = rays['rays_d']  # both (N_rays, 3)
    near = rays['near']
    far = rays['far']  # both (N_rays, 1)
    '''
        这行代码的意思是获取输入中光线方向数组 rays_d 的第一维的大小，
        也就是光线的数量。例如，如果 rays_d 是一个形状为 (1000, 3) 的数组，
        这意味着有 1000 条光线，每条光线有一个3维的方向向量，那么 N_rays 就会被设置为 1000。
    '''
    N_rays = rays_d.shape[0]

    # Embed direction
    # 方向嵌入: 将方向向量规范化后，使用方向嵌入模型进行嵌入。
    '''
    这行代码执行了两个操作：
    rays_d.norm(2,-1): 计算每条光线方向向量的 L2 范数（欧几里得范数），即求得光线方向向量的长度。
    rays_d / rays_d.norm(2,-1)[:,None]: 将每条光线方向向量除以其长度进行归一化，
    这样每个向量的长度都将是 1。[:,None] 是一种增加维度的操作，它将原来的向量长度变成一个 
    1×n 的二维数组，使得可以广播到每个光线方向向量上。

    代码 rays_d_norm = rays_d / rays_d.norm(2,-1)[:,None] 的解释:
    这行代码执行了光线方向的标准化。rays_d 是一个包含光线方向向量的张量。.norm(2,-1) 计算了每个方向向量的2范数（即欧几里得长度），并且指定 -1 作为维度参数意味着在最后一个维度上进行计算（这通常是每个向量的维度）。结果是一个长度为 (N_rays, ) 的向量，其中每个条目是对应光线方向的长度。通过将 rays_d 的每个方向向量除以其长度，我们得到一个单位长度（长度为1）的方向向量集合。[:,None] 是一个确保除法是在正确的维度上进行广播的技巧。


    “方向向量都有相同的长度（标准化为 1）” 的含义:
    当我们说一个向量被标准化了，意味着它的长度（或者说它的范数）被缩放到了1。
    在3D图形和光线追踪中，经常需要使用单位方向向量，因为它们表示了方向而不是长度。在这种情况下，所有的光线方向向量都被缩放为长度为1，这样每个向量仅仅表示方向，而与其原始的长度无关。这是计算和比较方向时的一个常见做法，可以简化后续计算。
    '''
    rays_d_norm = rays_d / rays_d.norm(2,-1)[:,None]

    '''
    这行代码表示的是将光线方向的单位向量进行嵌入（embedding），转换为一个高维空间中的向量。
    在NeRF中，方向嵌入有助于模型理解物体表面的方向性质，如光泽和阴影。
    嵌入模型通常是一个训练好的神经网络，它可以将3维的方向向量转换为一个更高维度的向量，
    这个向量携带了关于方向的更丰富信息。在代码中，embedding_dir 是一个嵌入模型，rays_d_norm 是单位化后的光线方向向量，
    dir_embedded 是嵌入后的方向向量。
    '''
    dir_embedded = embedding_dir(rays_d_norm) # (N_rays, embed_dir_channels)
    
    '''
    “深度值”（depth values）通常指的是从相机或观察点到场景中某点的距离。
    在3D计算机图形学和视觉中，深度值用于确定对象的远近，以正确渲染3D场景。
    例如，在一个3D场景中，摄像机到一个物体表面点的直线距离就是那个点的深度值。

    “深度空间”（depth space）指的是一个以深度值为基础的坐标系，在这个坐标系中，
    值通常是线性分布的，表示从近处到远处的范围。
    在渲染光线时，沿着每条光线在深度空间中均匀采样点，可以得到不同深度上的颜色和密度信息，以此来重建整个场景。
    例如，我们可以在一个线性的深度空间中采样，这意味着从近平面到远平面的每一个采样点都是均匀分布的。
    在另一个例子中，我们可能会在视差空间（disparity space）中采样，这时候采样点更加集中在近平面，
    因为视差变化在物体靠近摄像机时更加明显。       

    '''

    # Sample depth points
    # 深度点采样: 在 [0, 1] 范围内均匀采样 N_samples 次，根据是否使用视差空间采样来确定 z 值。
    '''
    深度采样点:
    深度采样点是指在 3D 空间中沿着一条光线方向的不同深度上的点。
    举例来说，如果你有一个从相机出发的光线，这条光线会穿过场景并与物体表面相交。
    在这条光线上，你可以选择一系列的点来代表不同的深度，这些点可以是均匀分布的，
    也可以根据一些启发式方法或其他信息（如粗糙模型的权重）来非均匀分布。这些点的集合就构成了深度采样点。
    例如，如果你在近平面（距离相机 1 米）和远平面（距离相机 10 米）之间进行采样，
    且 N_samples 为 5，那么深度采样点可能位于 1 米、3 米、5 米、7 米和 9 米处，假设是均匀采样的情况。


    '''
    '''
    这行代码生成了一个在区间 [0, 1] 内均匀分布的数字序列，总共 N_samples 个样本点。
    这个序列代表沿着每条光线的采样点在光线的“深度”上的相对位置，其中 0 表示最近的深度点，1 表示最远的深度点。
    torch.linspace 是 PyTorch 中的一个函数，用于生成在指定区间内均匀间隔的数字。
    device=rays_d.device 确保生成的这个张量在同一个设备上（CPU或GPU），与光线的方向张量 rays_d 在同一个设备上。

    视差空间采样和线性空间采样的区别：
    线性空间采样：在这种方式下，深度值是在 near 和 far（光线的近平面和远平面深度值）之间线性插值得到的。
    这意味着在近平面和远平面之间的深度值是均匀分布的。
    视差空间采样：这种采样方法基于视差，即物体在视网膜上的投影点之间的距离。
    在视差空间中采样时，深度值的分布是非线性的，接近观察者的地方采样点更密集，远离观察者的地方采样点更稀疏。
    这种方法通常用于处理透视效果，因为在透视视图中，物体看起来随着距离的增加而变小。
    在 render_rays 函数中，这两种采样方式都被考虑，通过 use_disp 参数来选择。
    视差采样通常在处理有透视效果的场景时更为有效，因为它能更好地模拟光线在不同深度的行为。     
    '''
    z_steps = torch.linspace(0, 1, N_samples, device=rays_d.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    '''
    'z_vals = z_vals.expand(N_rays, N_samples)' 这行代码的含义是将 z_vals 向量扩展到 (N_rays, N_samples) 的形状。
    z_vals 向量原本的形状是 (N_samples,)，代表在每条光线上采样深度的位置。
    这一步通过 .expand() 操作将 z_vals 复制到每一条光线上，以便能够为每一条光线的每一个采样点提供一个深度值。
    扩展操作并不复制数据，而是创建一个新的视图，其中每一条光线的 z_vals 都是相同的，这样可以节省内存并提高效率。
    举个例子，如果你有一个 (4,) 形状的向量 [a, b, c, d]，使用 .expand(2, 4) 操作后，
    你会得到一个 (2, 4) 形状的矩阵，两行都是 [a, b, c, d]，但这些数据并没有被复制两次，只是在内存中引用了原始向量。
    在这个函数中，N_rays 代表光线的数量，N_samples 代表每条光线上采样点的数量。这样每条光线都会有相同数量的采样深度，
    但每条光线上的采样点是独立处理的。
    '''
    z_vals = z_vals.expand(N_rays, N_samples)
    
    # 采样点扰动: 如果 perturb 大于 0，将采样深度（z 值）进行扰动，以增加渲染时的随机性。
    '''
    扰动采样点（Perturbing sampling points）是在神经辐射场（NeRF）和其他体积渲染技术中常用的一种技巧，
    目的是为了在渲染过程中引入随机性。这种随机性可以帮助模型更好地泛化，防止过拟合，并且可以产生更自然的渲染效果。
    在NeRF的上下文中，光线（rays）是从摄像机出发并穿过场景的虚拟线。为了渲染这些光线，我们需要在每条光线上选择一系列点，
    然后在这些点上估计体积密度（sigma）和颜色（rgb）。采样点通常是在光线的近平面和远平面之间均匀选择的。
    然而，如果这些点总是在相同的位置上均匀采样，那么模型可能会过于依赖于这些固定的采样位置，并且可能会忽视其他潜在重要的区域。
    为了避免这种情况，可以在每个采样点的位置上引入一些随机扰动。这种扰动是通过在采样点的位置上加上一些小的随机偏移来实现的。
    在代码中，这种扰动是通过以下步骤实现的：
    计算相邻采样点之间的中点（z_vals_mid）。
    在每对中点之间生成一个均匀分布的随机数（perturb_rand），这个随机数被乘以一个扰动因子（perturb）。
    使用这个随机数来在原来的采样间隔内选择一个新的深度值。
    通过这种方式，原本均匀的采样点就被随机扰动了，这样模型就被迫学习从不同位置采样时场景的多样性，从而增强了模型的泛化能力。   
    '''
    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)
        
        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays_d.device)
        z_vals = lower + (upper - lower) * perturb_rand

    # zvals are not optimized
    # produce points in the root body space
    '''
    生成采样点的空间位置:
    xyz_sampled 是根据光线原点 (rays_o)、
    光线方向 (rays_d) 和扰动后的深度值 (z_vals) 计算出的三维空间中的采样点坐标。
    '''
    xyz_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2) # (N_rays, N_samples, 3)

    if use_fine: # sample points for fine model
        # output: 
        #  loss:   'img_coarse', 'sil_coarse', 'feat_err', 'proj_err' 
        #               'vis_loss', 'flo/fdp_coarse', 'flo/fdp_valid',  
        #  not loss:   'depth_rnd', 'pts_pred', 'pts_exp'
        with torch.no_grad():
            _, weights_coarse = inference_deform(xyz_sampled, rays, models, 
                              chunk, N_samples,
                              N_rays, embedding_xyz, rays_d, noise_std,
                              obj_bound, dir_embedded, z_vals,
                              img_size, progress,opts,fine_iter=False)

        # reset N_importance
        N_importance = N_samples
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) 
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb==0)).detach()
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)

        N_samples = N_samples + N_importance # get back to original # of samples
    
    result, _ = inference_deform(xyz_sampled, rays, models, 
                          chunk, N_samples,
                          N_rays, embedding_xyz, rays_d, noise_std,
                          obj_bound, dir_embedded, z_vals,
                          img_size, progress,opts,render_vis=render_vis)

    return result



def inference(models, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code=None, weights_only=False, clip_bound = None, vis_pred=None):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine)
        embedding_xyz: embedding module for xyz
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model
        dir_: (N_rays, 3) ray directions
        dir_embedded: (N_rays, embed_dir_channels) embedded directions
        z_vals: (N_rays, N_samples_) depths of the sampled positions
        weights_only: do inference on sigma only or not

    Outputs:
        rgb_final: (N_rays, 3) the final rgb image
        depth_final: (N_rays) depth map
        weights: (N_rays, N_samples_): weights of each sample
    """
    nerf_sdf = models['coarse']
    N_samples_ = xyz_.shape[1]
    # Embed directions
    xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3)
    if not weights_only:
        dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                       # (N_rays*N_samples_, embed_dir_channels)

    # Perform model inference to get rgb and raw sigma
    chunk_size=4096
    B = xyz_.shape[0]
    xyz_input = xyz_.view(N_rays,N_samples,3)
    out = evaluate_mlp(nerf_sdf, xyz_input, 
            embed_xyz = embedding_xyz,
            dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
            code=env_code,
            chunk=chunk_size, sigma_only=weights_only).view(B,-1)

    rgbsigma = out.view(N_rays, N_samples_, 4)
    rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)
    sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)

    if 'nerf_feat' in models.keys():
        nerf_feat = models['nerf_feat']
        feat = evaluate_mlp(nerf_feat, xyz_input,
            embed_xyz = embedding_xyz,
            chunk=chunk_size).view(N_rays,N_samples_,-1)
    else:
        feat = torch.zeros_like(rgbs)

    # Convert these values using volume rendering (Section 4)
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
    # a hacky way to ensures prob. sum up to 1     
    # while the prob. of last bin does not correspond with the values
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

    noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

    # compute alpha by the formula (3)
    sigmas = sigmas+noise
    #sigmas = F.softplus(sigmas)
    #sigmas = torch.relu(sigmas)
    ibetas = 1/(nerf_sdf.beta.abs()+1e-9)
    #ibetas = 100
    sdf = -sigmas
    sigmas = (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * ibetas)) # 0-1
    # alternative: 
    #sigmas = F.sigmoid(-sdf*ibetas)
    sigmas = sigmas * ibetas

    alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i

    #set out-of-bound and nonvisible alphas to zero
    if clip_bound is not None:
        clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
        oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
        alphas[oob]=0
    if vis_pred is not None:
        alphas[vis_pred<0.5] = 0

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]
    weights = alphas * alpha_prod # (N_rays, N_samples_)
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    visibility = alpha_prod.detach() # 1 q_0 q_j-1

    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    feat_final = torch.sum(weights.unsqueeze(-1)*feat, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

    return rgb_final, feat_final, depth_final, weights, visibility
    
def inference_deform(xyz_coarse_sampled, rays, models, chunk, N_samples,
                         N_rays, embedding_xyz, rays_d, noise_std,
                         obj_bound, dir_embedded, z_vals,
                         img_size, progress,opts, fine_iter=True, 
                         render_vis=False):
    """
    fine_iter: whether to render loss-related terms
    render_vis: used for novel view synthesis
    """
    is_training = models['coarse'].training
    xys = rays['xys']

    # root space point correspondence in t2
    if opts.dist_corresp:
        xyz_coarse_target = xyz_coarse_sampled.clone()
        xyz_coarse_dentrg = xyz_coarse_sampled.clone()
    xyz_coarse_frame  = xyz_coarse_sampled.clone()

    # free deform
    if 'flowbw' in models.keys():
        model_flowbw = models['flowbw']
        model_flowfw = models['flowfw']
        time_embedded = rays['time_embedded'][:,None]
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
        flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, 
                             chunk=chunk//N_samples, xyz=xyz_coarse_sampled, code=time_embedded)
        xyz_coarse_sampled=xyz_coarse_sampled + flow_bw
       
        if fine_iter:
            # cycle loss (in the joint canonical space)
            xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
            flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                  chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded)
            frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
            # rigidity loss
            frame_disp3d = flow_fw.norm(2,-1)

            if "time_embedded_target" in rays.keys():
                time_embedded_target = rays['time_embedded_target'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_target)
                xyz_coarse_target=xyz_coarse_sampled + flow_fw
            
            if "time_embedded_dentrg" in rays.keys():
                time_embedded_dentrg = rays['time_embedded_dentrg'][:,None]
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_dentrg)
                xyz_coarse_dentrg=xyz_coarse_sampled + flow_fw


    elif 'bones' in models.keys():
        bones_rst = models['bones_rst']
        bone_rts_fw = rays['bone_rts']
        skin_aux = models['skin_aux']
        rest_pose_code =  models['rest_pose_code']
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        
        if 'nerf_skin' in models.keys():
            # compute delta skinning weights of bs, N, B
            nerf_skin = models['nerf_skin'] 
        else:
            nerf_skin = None
        time_embedded = rays['time_embedded'][:,None]
        # coords after deform
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True)
        skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                    bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)

        # backward skinning
        xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                  bone_rts_fw, 
                                                  skin_backward,
                                                  xyz_coarse_sampled,
                                                  )

        if fine_iter:
            #if opts.dist_corresp:
            skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                        bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)

            # cycle loss (in the joint canonical space)
            xyz_coarse_frame_cyc,_ = lbs(bones_rst, bone_rts_fw,
                              skin_forward, xyz_coarse_sampled, backward=False)
            frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)
            
            # rigidity loss (not used as optimization objective)
            num_bone = bones_rst.shape[0] 
            bone_fw_reshape = bone_rts_fw.view(-1,num_bone,12)
            bone_trn = bone_fw_reshape[:,:,9:12]
            bone_rot = bone_fw_reshape[:,:,0:9].view(-1,num_bone,3,3)
            frame_rigloss = bone_trn.pow(2).sum(-1)+rot_angle(bone_rot)
            
            if opts.dist_corresp and 'bone_rts_target' in rays.keys():
                bone_rts_target = rays['bone_rts_target']
                xyz_coarse_target,_ = lbs(bones_rst, bone_rts_target, 
                                   skin_forward, xyz_coarse_sampled,backward=False)
            if opts.dist_corresp and 'bone_rts_dentrg' in rays.keys():
                bone_rts_dentrg = rays['bone_rts_dentrg']
                xyz_coarse_dentrg,_ = lbs(bones_rst, bone_rts_dentrg, 
                                   skin_forward, xyz_coarse_sampled,backward=False)

    # nerf shape/rgb
    model_coarse = models['coarse']
    if 'env_code' in rays.keys():
        env_code = rays['env_code']
    else:
        env_code = None

    # set out of bounds weights to zero
    if render_vis: 
        clip_bound = obj_bound
        xyz_embedded = embedding_xyz(xyz_coarse_sampled)
        vis_pred = evaluate_mlp(models['nerf_vis'], 
                               xyz_embedded, chunk=chunk)[...,0].sigmoid()
    else:
        clip_bound = None
        vis_pred = None


    if opts.symm_shape:
        ##TODO set to x-symmetric here
        symm_ratio = 0.5
        xyz_x = xyz_coarse_sampled[...,:1].clone()
        symm_mask = torch.rand_like(xyz_x) < symm_ratio
        xyz_x[symm_mask] = -xyz_x[symm_mask]
        xyz_input = torch.cat([xyz_x, xyz_coarse_sampled[...,1:3]],-1)
    else:
        xyz_input = xyz_coarse_sampled

    rgb_coarse, feat_rnd, depth_rnd, weights_coarse, vis_coarse = \
        inference(models, embedding_xyz, xyz_input, rays_d,
                dir_embedded, z_vals, N_rays, N_samples, chunk, noise_std,
                weights_only=False, env_code=env_code, 
                clip_bound=clip_bound, vis_pred=vis_pred)
    sil_coarse =  weights_coarse[:,:-1].sum(1)
    result = {'img_coarse': rgb_coarse,
              'depth_rnd': depth_rnd,
              'sil_coarse': sil_coarse,
             }
  
    # render visibility scores
    if render_vis:
        result['vis_pred'] = (vis_pred * weights_coarse).sum(-1)

    if fine_iter:
        if opts.use_corresp:
            # for flow rendering
            pts_exp = compute_pts_exp(weights_coarse, xyz_coarse_sampled)
            pts_target = kp_reproj(pts_exp, models, embedding_xyz, rays, 
                                to_target=True) # N,1,2
        # viser feature matching
        if 'feats_at_samp' in rays.keys():
            feats_at_samp = rays['feats_at_samp']
            nerf_feat = models['nerf_feat']
            xyz_coarse_sampled_feat = xyz_coarse_sampled
            weights_coarse_feat = weights_coarse
            pts_pred, pts_exp, feat_err = feat_match_loss(nerf_feat, embedding_xyz,
                       feats_at_samp, xyz_coarse_sampled_feat, weights_coarse_feat,
                       obj_bound, is_training=is_training)


            # 3d-2d projection
            proj_err = kp_reproj_loss(pts_pred, xys, models, 
                    embedding_xyz, rays)
            proj_err = proj_err/img_size * 2
            
            result['pts_pred'] = pts_pred
            result['pts_exp']  = pts_exp
            result['feat_err'] = feat_err # will be used as loss
            result['proj_err'] = proj_err # will be used as loss

        if opts.dist_corresp and 'rtk_vec_target' in rays.keys():
            # compute correspondence: root space to target view space
            # RT: root space to camera space
            rtk_vec_target =  rays['rtk_vec_target']
            Rmat = rtk_vec_target[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_target[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_target[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_target = obj_to_cam(xyz_coarse_target, Rmat, Tmat) 
            xyz_coarse_target = pinhole_cam(xyz_coarse_target,K)

        if opts.dist_corresp and 'rtk_vec_dentrg' in rays.keys():
            # compute correspondence: root space to dentrg view space
            # RT: root space to camera space
            rtk_vec_dentrg =  rays['rtk_vec_dentrg']
            Rmat = rtk_vec_dentrg[:,0:9].view(N_rays,1,3,3)
            Tmat = rtk_vec_dentrg[:,9:12].view(N_rays,1,3)
            Kinv = rtk_vec_dentrg[:,12:21].view(N_rays,1,3,3)
            K = mat2K(Kmatinv(Kinv))

            xyz_coarse_dentrg = obj_to_cam(xyz_coarse_dentrg, Rmat, Tmat) 
            xyz_coarse_dentrg = pinhole_cam(xyz_coarse_dentrg,K)
        
        # raw 3d points for visualization
        result['xyz_camera_vis']   = xyz_coarse_frame 
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            result['xyz_canonical_vis']   = xyz_coarse_sampled
        if 'feats_at_samp' in rays.keys():
            result['pts_exp_vis']   = pts_exp
            result['pts_pred_vis']   = pts_pred
            
        if 'flowbw' in models.keys() or  'bones' in models.keys():
            # cycle loss (in the joint canonical space)
            #if opts.dist_corresp:
            result['frame_cyc_dis'] = (frame_cyc_dis * weights_coarse.detach()).sum(-1)
            #else:
            #    pts_exp_reg = pts_exp[:,None].detach()
            #    skin_forward = gauss_mlp_skinning(pts_exp_reg, embedding_xyz, 
            #                bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)
            #    pts_exp_fw,_ = lbs(bones_rst, bone_rts_fw,
            #                      skin_forward, pts_exp_reg, backward=False)
            #    skin_backward = gauss_mlp_skinning(pts_exp_fw, embedding_xyz, 
            #                bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)
            #    pts_exp_fwbw,_ = lbs(bones_rst, bone_rts_fw,
            #                       skin_backward,pts_exp_fw)
            #    frame_cyc_dis = (pts_exp_fwbw - pts_exp_reg).norm(2,-1)
            #    result['frame_cyc_dis'] = sil_coarse.detach() * frame_cyc_dis[...,-1]
            if 'flowbw' in models.keys():
                result['frame_rigloss'] =  (frame_disp3d  * weights_coarse.detach()).sum(-1)
                # only evaluate at with_grad mode
                if xyz_coarse_frame.requires_grad:
                    # elastic energy
                    result['elastic_loss'] = elastic_loss(model_flowbw, embedding_xyz, 
                                      xyz_coarse_frame, time_embedded)
            else:
                result['frame_rigloss'] =  (frame_rigloss).mean(-1)

            
            ### script to plot sigmas/weights
            #from matplotlib import pyplot as plt
            #plt.ioff()
            #sil_rays = weights_coarse[rays['sil_at_samp'][:,0]>0]
            #plt.plot(sil_rays[::1000].T.cpu().numpy(),'*-')
            #plt.savefig('tmp/probs.png')
            #plt.cla()

        if is_training and 'nerf_vis' in models.keys():
            result['vis_loss'] = visibility_loss(models['nerf_vis'], embedding_xyz,
                            xyz_coarse_sampled, vis_coarse, obj_bound, chunk)

        # render flow 
        if 'rtk_vec_target' in rays.keys():
            if opts.dist_corresp:
                flo_coarse, flo_valid = vrender_flo(weights_coarse, xyz_coarse_target,
                                                    xys, img_size)
            else:
                flo_coarse = diff_flo(pts_target, xys, img_size)
                flo_valid = torch.ones_like(flo_coarse[...,:1])

            result['flo_coarse'] = flo_coarse
            result['flo_valid'] = flo_valid

        if 'rtk_vec_dentrg' in rays.keys():
            if opts.dist_corresp:
                fdp_coarse, fdp_valid = vrender_flo(weights_coarse, 
                                                    xyz_coarse_dentrg, xys, img_size)
            else:
                fdp_coarse = diff_flo(pts_dentrg, xys, img_size)
                fdp_valid = torch.ones_like(fdp_coarse[...,:1])
            result['fdp_coarse'] = fdp_coarse
            result['fdp_valid'] = fdp_valid

        if 'nerf_unc' in models.keys():
            # xys: bs,nsample,2
            # t: bs
            nerf_unc = models['nerf_unc']
            ts = rays['ts']
            vid_code = rays['vid_code']

            # change according to K
            xysn = rays['xysn']
            xyt = torch.cat([xysn, ts],-1)
            xyt_embedded = embedding_xyz(xyt)
            xyt_code = torch.cat([xyt_embedded, vid_code],-1)
            unc_pred = nerf_unc(xyt_code)
            #TODO add activation function
            #unc_pred = F.softplus(unc_pred)
            result['unc_pred'] = unc_pred
        
        if 'img_at_samp' in rays.keys():
            # compute other losses
            img_at_samp = rays['img_at_samp']
            sil_at_samp = rays['sil_at_samp']
            vis_at_samp = rays['vis_at_samp']
            flo_at_samp = rays['flo_at_samp']
            cfd_at_samp = rays['cfd_at_samp']

            # img loss
            img_loss_samp = (rgb_coarse - img_at_samp).pow(2).mean(-1)[...,None]
            
            # sil loss, weight sil loss based on # points
            if is_training and sil_at_samp.sum()>0 and (1-sil_at_samp).sum()>0:
                pos_wt = vis_at_samp.sum()/   sil_at_samp[vis_at_samp>0].sum()
                neg_wt = vis_at_samp.sum()/(1-sil_at_samp[vis_at_samp>0]).sum()
                sil_balance_wt = 0.5*pos_wt*sil_at_samp + 0.5*neg_wt*(1-sil_at_samp)
            else: sil_balance_wt = 1
            sil_loss_samp = (sil_coarse[...,None] - sil_at_samp).pow(2) * sil_balance_wt
            sil_loss_samp = sil_loss_samp * vis_at_samp
               
            # flo loss, confidence weighting: 30x normalized distance - 0.1x pixel error
            flo_loss_samp = (flo_coarse - flo_at_samp).pow(2).sum(-1)
            # hard-threshold cycle error
            sil_at_samp_flo = (sil_at_samp>0)\
                     & (flo_valid==1)
            sil_at_samp_flo[cfd_at_samp==0] = False 
            if sil_at_samp_flo.sum()>0:
                cfd_at_samp = cfd_at_samp / cfd_at_samp[sil_at_samp_flo].mean()
            flo_loss_samp = flo_loss_samp[...,None] * cfd_at_samp
       
            result['img_at_samp']   = img_at_samp
            result['sil_at_samp']   = sil_at_samp
            result['vis_at_samp']   = vis_at_samp
            result['sil_at_samp_flo']   = sil_at_samp_flo
            result['flo_at_samp']   = flo_at_samp
            result['img_loss_samp'] = img_loss_samp 
            result['sil_loss_samp'] = sil_loss_samp
            result['flo_loss_samp'] = flo_loss_samp
    
            # exclude error outside mask
            result['img_loss_samp']*=sil_at_samp
            result['flo_loss_samp']*=sil_at_samp

        if 'feats_at_samp' in rays.keys():
            # feat loss
            feats_at_samp=rays['feats_at_samp']
            feat_rnd = F.normalize(feat_rnd, 2,-1)
            frnd_loss_samp = (feat_rnd - feats_at_samp).pow(2).mean(-1)
            result['frnd_loss_samp'] = frnd_loss_samp * sil_at_samp[...,0]
    return result, weights_coarse


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


