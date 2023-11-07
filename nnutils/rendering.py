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
        #  not loss:   'depth_rnd', 'pts_pred', 'pts_exp'、
        '''
        为细致模型采样点:
        如果 use_fine 为真，表示需要进行细致模型的渲染。
        首先，使用粗略模型的结果（weights_coarse）来决定在哪些区域需要更密集的采样（这是所谓的重要性采样）。
        sample_pdf 函数基于粗略模型的权重来采样额外的深度值，这些值后续会被用于细致模型的渲染。
        新采样的深度值 (z_vals_) 被合并到原始的深度值中，并且排序，以便用于细致模型的渲染。 
        '''
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
    

    '''
    渲染结果:
    最后，调用 inference_deform 函数使用采样点的坐标 (xyz_sampled) 和光线信息来计算最终的渲染结果。
    '''

    result, _ = inference_deform(xyz_sampled, rays, models, 
                          chunk, N_samples,
                          N_rays, embedding_xyz, rays_d, noise_std,
                          obj_bound, dir_embedded, z_vals,
                          img_size, progress,opts,render_vis=render_vis)
    
    # 函数返回一个结果字典，它包含了粗略和细致模型的颜色和深度信息。
    return result
    '''
    整个过程涉及到两次采样：一次是初始的均匀采样，另一次是基于粗略模型权重的重要性采样。
    这种方法可以提高渲染质量，因为它允许模型在视觉上更复杂的区域分配更多的计算资源。
    '''

# 为什么叫'inference':
# 'Inference'在机器学习和统计领域指的是根据已训练好的模型和给定的输入数据预测输出结果的过程。

def inference(models, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, 
        N_rays, N_samples,chunk, noise_std,
        env_code=None, weights_only=False, clip_bound = None, vis_pred=None):
    """
    Helper function that performs model inference.

    Inputs:
        model: NeRF model (coarse or fine) 包含NeRF模型的字典，通常有粗糙（coarse）和细腻（fine）两种。
        embedding_xyz: embedding module for xyz 空间位置的嵌入模型。
        xyz_: (N_rays, N_samples_, 3) sampled positions
              N_samples_ is the number of sampled points in each ray;
                         = N_samples for coarse model
                         = N_samples+N_importance for fine model 采样的空间位置点，维度为(N_rays, N_samples_, 3)
        dir_: (N_rays, 3) ray directions  射线的方向，维度为(N_rays, 3)
        dir_embedded: (N_rays, embed_dir_channels) embedded directions 方向的嵌入表示，维度为(N_rays, embed_dir_channels)
        z_vals: (N_rays, N_samples_) depths of the sampled positions 采样点的深度值，维度为(N_rays, N_samples_)
        weights_only: do inference on sigma only or not  布尔值，指示是否仅对密度（sigma）进行推理。

    Outputs:
        rgb_final: (N_rays, 3) the final rgb image
        depth_final: (N_rays) depth map
        weights: (N_rays, N_samples_): weights of each sample
    """

    '''
    函数用途:
    这个函数负责根据输入的空间位置（xyz_）、方向（dir_和dir_embedded）和深度值（z_vals），通过NeRF模型来预测每个采样点的颜色（RGB值）和透明度（或称为密度）。然后使用体积渲染公式来合成整条射线上的颜色，从而得到最终图像的颜色和深度图。例如，假设你有一个3D场景，你想渲染一个视角下的图像，你会沿着视角向场景内部发射许多射线，对于每条射线，这个函数计算它通过场景时的颜色变化。
    
    'dir_'和'dir_embedded'的区别:
    dir_ 是射线的方向向量，而 dir_embedded 是经过嵌入（embedding）网络处理后的射线方向。嵌入网络通常使用一些高维变换来增加数据的表现力，以便NeRF模型能更好地解释方向对光线颜色的影响。
    
    函数 torch.repeat_interleave() 在这里被用于创建一个与采样点数量相匹配的方向向量 dir_embedded 的副本，因为每个采样点都需要与一个方向向量相对应来计算其颜色值。

    dir_ 是原始的射线方向向量，而 dir_embedded 是经过嵌入网络处理后的方向向量，这种嵌入通常是一种高维表示，它捕捉了方向的细微变化，这在神经辐射场模型中是重要的。这个嵌入的方向向量 dir_embedded 将与空间位置 xyz_ 结合，一起输入到 NeRF 模型中以计算颜色。

    在推理过程中，每个射线上有多个采样点，每个采样点都需要一个方向向量来计算其颜色。因此，对于射线上的每个采样点，我们都需要复制 dir_embedded。dir_ 本身并不需要复制，因为它仅表示射线的原始方向，并不直接用于模型的颜色计算。
    
    '''
    nerf_sdf = models['coarse']     # 从模型字典中获取粗糙模型。
    # xyz_ 是沿射线采样的三维空间点的坐标，形状为 (N_rays, N_samples_, 3)，
    # 其中 N_rays 是射线的数量，N_samples_ 是每条射线上采样点的数量。
    N_samples_ = xyz_.shape[1]      # 获取每条射线的采样点数。
    # Embed directions
    xyz_ = xyz_.view(-1, 3) # (N_rays*N_samples_, 3) 将采样点重塑为合适的形状以进行推理。
    '''
    这行代码 dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0) 的意思是
    将嵌入后的方向向量 dir_embedded 沿着第一个维度（射线的数量）重复 N_samples_ 次。
    N_samples_ 是每条射线上采样的点数。这样做的目的是为了匹配位置点 xyz_ 的数量，
    因为每个位置点都需要一个对应的方向向量来计算颜色。如果 weights_only 为 False，
    表示不仅需要计算透明度权重，还需要计算颜色，所以需要重复方向向量以便于后续的计算。

    为什么要重复方向向量？
    因为 NeRF 模型需要对每个采样点的位置和方向进行推理。在模型中，每个采样点的颜色可能会因为射线的方向不同而有所不同，所以需要对每个点指定一个方向向量。由于在一条射线上有多个采样点，但它们共享同一条射线的方向，所以需要将射线的方向向量复制多次，以确保每个采样点都有一个方向向量与之对应。
    
    需要重复的次数为采样点数，对吗？
    是的，重复的次数应该等于每条射线上采样点的数量。这是因为在计算颜色时，每个采样点都需要一个对应的方向向量。

    为什么计算颜色需要方向向量？
    计算颜色需要方向向量，因为在真实世界中，一个物体表面的颜色取决于光线照射的方向。
    这种现象称为双向反射分布函数（BRDF），它描述了光如何从表面散射。
    在 NeRF 中，需要模拟这种效果，因为光线从不同方向照射到同一点上时，观察到的颜色可能会有所不同。
    因此，渲染过程必须考虑每个点的观察方向，以正确地估计最终颜色。

    这个条件检查是否在执行模型推理时需要计算除了透明度权重（sigma）之外的其他输出（如颜色rgb）。
    如果 weights_only 是 False，那么这个条件为真，意味着需要计算颜色和透明度权重；
    如果 weights_only 是 True，那么这个条件为假，意味着只计算透明度权重。

    torch.repeat_interleave() 是 PyTorch 中的一个函数，它将输入张量的元素沿指定的维度重复多次。这个函数的输入参数包括：

    第一个参数是要重复的张量。
    repeats 是一个整数或者整数的张量，它指定每个元素要重复的次数。
    dim 是可选的，指定要沿着哪个维度进行重复。如果不指定，则会扁平化输入并返回一个一维的结果。

    '''
    if not weights_only:            
        dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
                       # (N_rays*N_samples_, embed_dir_channels)

    # Perform model inference to get rgb and raw sigma
    chunk_size=4096     # 设置批量处理的块大小。
    B = xyz_.shape[0]   # 计算批处理的总数。
    xyz_input = xyz_.view(N_rays,N_samples,3)   # 重新组织输入数据。
    # 函数进行模型推理，计算rgb值和原始sigma。
    '''
    evaluate_mlp 函数负责处理配对功能。当 xyz_input 和 dir_embedded 被送入 evaluate_mlp 函数时，
    由于它们的形状都是 (N_rays, N_samples, -1)，这个函数隐含地对每一对位置和方向嵌入向量进行操作。

    evaluate_mlp 函数会接受这些输入，然后对每一组位置向量和方向嵌入向量进行评估，
    得到对应的颜色值和密度值。因为这两个输入已经具有相同的第一维度（即光线的数量）和第二维度（即每条光线上的样本数量），
    所以模型会对每一对 (xyz, dir) 进行计算。这就是如何确保每个位置点的计算都与相应的方向信息配对的。
    
    '''
    out = evaluate_mlp(nerf_sdf, xyz_input,     
            embed_xyz = embedding_xyz,
            dir_embedded = dir_embedded.view(N_rays,N_samples,-1),
            code=env_code,
            chunk=chunk_size, sigma_only=weights_only).view(B,-1)

    rgbsigma = out.view(N_rays, N_samples_, 4)  # 将输出重新组织为rgb和sigma。
    rgbs = rgbsigma[..., :3] # (N_rays, N_samples_, 3)  # 获取rgb值。
    sigmas = rgbsigma[..., 3] # (N_rays, N_samples_)    # # 获取sigma值。

    # 如果有特征模型，则进行推理
    if 'nerf_feat' in models.keys():
        nerf_feat = models['nerf_feat']
        feat = evaluate_mlp(nerf_feat, xyz_input,  
            embed_xyz = embedding_xyz,
            chunk=chunk_size).view(N_rays,N_samples_,-1)
    #  否则使用全零矩阵。
    else:
        feat = torch.zeros_like(rgbs)

    # Convert these values using volume rendering (Section 4)
    # 计算每对相邻采样点之间的距离
    '''
    计算每对相邻采样点之间的距离的原因:
    这个距离用于计算体渲染中每个采样点的贡献，包括颜色和密度的累积，这是体渲染算法的关键部分。
    '''
    deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1) 
    # a hacky way to ensures prob. sum up to 1     
    # while the prob. of last bin does not correspond with the values
    delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity  为最后一个采样点设置无限大的深度差。
    deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)
    
    # 计算实际世界距离并添加噪声。
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

    # alphas代表不透明度
    alphas = 1-torch.exp(-deltas*sigmas) # (N_rays, N_samples_), p_i

    # 设置边界条件和可见性：
    #set out-of-bound and nonvisible alphas to zero
    # oob: 如果采样点超出边界，则将其透明度设置为0。
    if clip_bound is not None:
        clip_bound = torch.Tensor(clip_bound).to(xyz_.device)[None,None]
        oob = (xyz_.abs()>clip_bound).sum(-1).view(N_rays,N_samples)>0
        alphas[oob]=0
    #  根据可见性预测来调整透明度。
    if vis_pred is not None:
        alphas[vis_pred<0.5] = 0

    alphas_shifted = \
        torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, a1, a2, ...]
    # 计算透明度的累积乘积。
    alpha_prod = torch.cumprod(alphas_shifted, -1)[:, :-1]
    # 计算每个采样点的权重
    weights = alphas * alpha_prod # (N_rays, N_samples_)
    # 计算沿射线的累积不透明度。
    weights_sum = weights.sum(1) # (N_rays), the accumulated opacity along the rays
                                 # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically
    # 计算可见性。
    visibility = alpha_prod.detach() # 1 q_0 q_j-1

    # 计算加权的最终输出，包括rgb图像、特征和深度图。
    # compute final weighted outputs
    rgb_final = torch.sum(weights.unsqueeze(-1)*rgbs, -2) # (N_rays, 3)
    feat_final = torch.sum(weights.unsqueeze(-1)*feat, -2) # (N_rays, 3)
    depth_final = torch.sum(weights*z_vals, -1) # (N_rays)

    return rgb_final, feat_final, depth_final, weights, visibility

    # 这个函数的核心是将空间点的颜色和密度值转换为一张图片，并返回对应的rgb值、特征、深度图以及每个采样点的权重和可见性。
    # 这是体积渲染算法的一个关键步骤，它允许NeRF模型生成具有深度感知的3D渲染图片。

def inference_deform(xyz_coarse_sampled, rays, models, chunk, N_samples,
                         N_rays, embedding_xyz, rays_d, noise_std,
                         obj_bound, dir_embedded, z_vals,
                         img_size, progress,opts, fine_iter=True, 
                         render_vis=False):
    """
    fine_iter: whether to render loss-related terms
    render_vis: used for novel view synthesis
    """
    '''
    输入：
    xyz_coarse_sampled 含义是：xyz: 指的是三维坐标系中的点，通常是 x, y, 和 z 坐标。
        coarse: 指的是这些点是用于粗略估计场景的第一步，NeRF模型通常有两个阶段：粗糙阶段和精细阶段。
        粗略阶段快速估计整个体积的密度和颜色，然后精细阶段会在重要区域进行更精细的采样和估计。
        sampled: 表示这些点是从沿射线的连续区域中采样得到的。
    rays: 各种射线和视图的参数集合,rays 参数可能包含如下信息：
        xys: 2D 像素坐标
        rays_d: 射线方向
        time_embedded: 用于处理动态场景的时间嵌入向量
        env_code 或 vid_code: 环境或视频的编码，用于条件渲染
        ts: 可能表示时间步长的参数
    models: 包含不同部分的 NeRF 模型的字典
    chunk: 用于计算的数据块的大小 参数 chunk 通常用于指定在单个批次中处理的射线数量或者采样点数量。
        这是为了在推理（或训练）过程中控制内存使用量，以免超出可用的计算资源，特别是在 GPU 上。
        由于 NeRF 模型通常需要对成千上万条射线进行采样和评估，这可能需要大量的内存。
        通过将这些射线分成更小的批次或“块”来处理，可以保证即使是具有有限内存的系统也能够处理这些计算。
        每个“块”包含了一定数量的射线或采样点，并且依次处理每个块，直到所有射线都被评估为止。
    N_samples: 每条射线上采样点的数量
    N_rays: 射线的数量
    embedding_xyz: 空间位置的嵌入函数, 它接受三维空间中的点坐标作为输入，并输出这些点的嵌入表示。
        这个嵌入表示然后被用作神经网络（比如 NeRF）的输入，以便进行场景的体积渲染。
        例如，对于 NeRF 应用，原始的x,y,z 坐标可能会通过傅里叶变换或其他映射变为一个更高维的空间，
        以帮助模型更好地学习和渲染复杂的场景细节。这种技术是提高渲染质量和模型性能的关键步骤之一。
    rays_d: 射线方向
    noise_std: 添加到体积密度预测中的噪声标准差
    obj_bound: 通常指的是用于确定场景或对象边界的参数。在三维渲染和图形中，obj_bound 可以用来指定一个空间区域，
        比如一个盒子（bounding box），它定义了你希望渲染或进行处理的区域的大小和位置。
        例如，在使用神经辐射场（NeRF）进行场景渲染时，obj_bound 可以用来指定场景的大小，
        确保渲染过程中只考虑这个空间区域内的点。这有助于优化性能，因为它允许模型忽略不在感兴趣区域外的点。
        在上下文中，obj_bound 可能是一个三元组或六元组，表示场景的最小和最大坐标。
        例如，如果场景被限制在一个单位立方体中，obj_bound 可能是 (-1, -1, -1, 1, 1, 1)，
        这表示对象在 x、y、z 轴上的边界分别在 -1 和 1 之间。
        在代码中，obj_bound 可能被用于以下几种情况：

        在渲染可见性时，确定哪些点应该被认为是在对象的边界内，哪些点应该被忽略。
        在体积渲染中，用于确定积分的终止条件，即在哪个点停止沿射线积分。
        在变形或动画中，确定哪些点可以移动，以及它们可以移动的范围。
        由于具体的实现细节可能会根据实际的应用场景和代码库的设计而有所不同，obj_bound 的确切含义和用法需要结合上下文来理解
    dir_embedded: 射线方向的嵌入表示
    z_vals: 沿射线的深度值
    img_size: 图像的大小
    progress: 渲染进度
    opts: 选项和配置
    fine_iter: 是否渲染与损失相关的项。当 fine_iter 为 True，函数会计算一些额外的信息，
        比如循环一致性（cycle consistency）损失和刚度（rigidity）损失，
        这些信息通常用于优化模型的表现。当 fine_iter 为 False，这些额外的计算可以被省略，
        这通常在纯粹的渲染或推理阶段，而非训练阶段发生。
        例子：
        假设我们正在训练一个用于渲染动态场景的 NeRF 模型。
        在每次训练迭代中，我们希望计算额外的损失项来引导模型更好地学习场景的动态变化。
        在这种情况下，我们可以将 fine_iter 设置为 True，样，inference_deform 函数会执行额外的损失相关计算，
        它们将用于反向传播和模型参数的更新。
        另一方面，如果我们已经完成了训练，现在只想用训练好的模型来渲染图像，我们可以将 fine_iter 设置为 False，
        在这种情况下，inference_deform 将跳过那些仅在训练时需要的计算，可能会导致渲染速度更快。


    render_vis: 是否用于新视图合成

    输出：
    result: 包含渲染结果和相关数据的字典
    weights_coarse: 粗糙渲染权重
    
    
    '''
    is_training = models['coarse'].training # 检查粗糙模型是否在训练模式。
    '''
    在 PyTorch 中，模型对象（通常是 nn.Module 的子类）具有 .training 属性，该属性是一个布尔值，
    表示模型当前是处于训练模式（.training == True）还是评估模式（.training == False）。
    在训练模式下，模型会记录梯度，并且某些层（如 Dropout 和 BatchNorm）会表现出与评估模式不同的行为。
    举个例子，假设我们有一个简单的神经网络模型，我们可以根据是否正在训练来改变其行为：
    在这个例子中，.train() 方法被用来将模型设置为训练模式，.eval() 方法被用来将模型设置为评估模式。
    根据 model.training 的值，dropout 层会在训练模式下随机“关闭”一些神经元以防止过拟合，在评估模式下则不会这样做。
    '''
    xys = rays['xys']   # 提取射线参数中的 xy 坐标。

    # 以下是处理点的变形和运动的代码块：
    # root space point correspondence in t2
    # 如果 opts.dist_corresp 为真，函数会克隆粗糙采样点的位置用于后续的变换计算。
    '''
    xyz_coarse_target = xyz_coarse_sampled.clone(): 
        创建 xyz_coarse_sampled 的一个完全独立的副本，并将其赋值给 xyz_coarse_target。
        这意味着后续的操作会在 xyz_coarse_target 上执行，而不会影响 xyz_coarse_sampled。
    xyz_coarse_dentrg = xyz_coarse_sampled.clone(): 
        同样创建了 xyz_coarse_sampled 的另一个副本，并赋值给 xyz_coarse_dentrg。
    xyz_coarse_frame = xyz_coarse_sampled.clone(): 
        再次复制 xyz_coarse_sampled，赋值给 xyz_coarse_frame。
    
    '''
    # 后续的变形操作可以分别应用于这三个副本上，而原始的 xyz_coarse_sampled 仍然保持不变，以便于比较和其他计算。
    # 可能是一个配置选项，指示是否需要距离对应关系
    if opts.dist_corresp:
        xyz_coarse_target = xyz_coarse_sampled.clone()
        xyz_coarse_dentrg = xyz_coarse_sampled.clone()
    xyz_coarse_frame  = xyz_coarse_sampled.clone()

    # free deform
    '''
    如果模型字典中包含 flowbw 键，表明要考虑背向流模型 model_flowbw 和前向流模型 model_flowfw。
    这部分代码通过这两个模型预测点的运动，以及实现循环一致性和刚度损失。
    '''

    '''
    这段代码处理的是一种称为“双向流”的变形过程，其中模型预测了从一个时间步到另一个时间步的点的运动。
    它使用了两个模型：model_flowbw（向后流）和model_flowfw（向前流），来估计点在时间维度上的位移。
    这通常用于处理动态场景，其中物体或相机之间的位置随时间变化。
    '''
    # 检查模型字典 models 是否包含键 flowbw，这意味着需要进行向后（backward）流预测。
    '''
    这段代码似乎是基于神经辐射场（NeRF）的扩展，用于处理动态场景。
    在NeRF的基础上，引入了时间维度的处理，以及对流动（或位移）的估计，这允许模型捕捉场景随时间的变化。
    具体来说，它使用神经网络来预测从一个时间步到另一个时间步的点的位移，这在处理视频或动态场景的NeRF应用中是非常重要的。
    
    '''
    if 'flowbw' in models.keys():
        # flowbw（向后流模型）和flowfw（向前流模型）通常是成对出现的，因为它们共同负责估计时间维度上的双向流动。
        model_flowbw = models['flowbw'] # 从字典中获取 model_flowbw，这是用于计算向后流的神经网络模型。
        model_flowfw = models['flowfw'] # 这是用于计算向前（forward）流的神经网络模型。
        '''
        [:, None] 是一种索引操作，它的作用是增加数组的一个维度。
        这种操作通常用于将一维数组转换为二维列向量。这里的 None 是一个内置常量，当它用在索引操作中时，
        它等同于 numpy.newaxis，其作用是在这个位置增加一个新的轴。
        rays 是一个字典，其中包含不同的射线（rays）相关数据，这里可能是与时间相关的信息。
        rays['time_embedded'] 通过键 'time_embedded' 从字典中获取时间嵌入向量，这是一个一维数组。
        [:, None] 被用来将 rays['time_embedded'] 转换为一个二维数组（列向量）。如果 rays['time_embedded'] 原本的形状是 (n,)，那么操作后的形状会变成 (n, 1)。
        '''
        time_embedded = rays['time_embedded'][:,None] # 这是表示时间信息的嵌入向量。
        xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled) # 对粗糙采样的点 xyz_coarse_sampled 进行空间嵌入。
        # 使用向后流模型 model_flowbw 评估流，预测每个点在时间维度上的位移
        '''
        model_flowbw：这是一个MLP神经网络模型，它被训练用来预测点云的时间维度上的位移。
        xyz_coarse_embedded：表示点云的空间位置的嵌入向量。
            在神经场景表示中，原始三维坐标通常被嵌入到一个高维空间以捕获更丰富的信息。
        chunk=chunk//N_samples：这指定了处理的批次大小。由于GPU的内存限制，通常需要将大量的点云分批处理。
            这里它将原始的批量大小 chunk 除以每条光线的样本数 
        N_samples，这可能是因为每个点都需要单独评估，所以总批次大小需要相应减小。
        xyz=xyz_coarse_sampled：这是实际的三维点云数据，表示点云在空间中的位置。
        code=time_embedded：这是时间信息的嵌入向量，它提供了模型预测位移的时间上下文。
        '''
        flow_bw = evaluate_mlp(model_flowbw, xyz_coarse_embedded, 
                             chunk=chunk//N_samples, xyz=xyz_coarse_sampled, code=time_embedded)
        xyz_coarse_sampled=xyz_coarse_sampled + flow_bw # 并将其加到原始采样点上，得到位移后的点 xyz_coarse_sampled。
       
        # 检查 fine_iter 是否为 True。这是一个条件，如果为 True，则执行以下的代码块。
        if fine_iter:
            # cycle loss (in the joint canonical space)
            # 将粗糙样本点 xyz_coarse_sampled 通过嵌入函数 embedding_xyz 进行转换，得到其嵌入表示 xyz_coarse_embedded。
            xyz_coarse_embedded = embedding_xyz(xyz_coarse_sampled)
            # 使用多层感知器 (MLP) 模型 model_flowfw 来评估前向流 flow_fw。这使用了嵌入的粗糙样本点，
            # 块大小（每块的数据量）由 chunk 除以样本数量 N_samples 决定，还需要原始的粗糙样本点和时间嵌入 time_embedded。
            flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                                  chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded)
            # 计算前向流 flow_fw 和一个未在代码中定义的后向流 flow_bw 的和，然后取其L2范数。这可能是为了计算一个循环损失，表示前向和后向流之间的一致性。
            frame_cyc_dis = (flow_bw+flow_fw).norm(2,-1)
            # 计算刚性损失（rigidity loss），即向前流的欧几里得范数，用于评估点位移的刚性。
            # rigidity loss
            # 计算前向流 flow_fw 的L2范数。这表示3D点的位移量。
            frame_disp3d = flow_fw.norm(2,-1)

            # 检查 rays 字典是否包含键 "time_embedded_target"。
            if "time_embedded_target" in rays.keys():
                # 如果上述条件满足，从 rays 字典中提取 "time_embedded_target" 并增加一个新的轴。
                time_embedded_target = rays['time_embedded_target'][:,None]
                # 重新评估前向流，但这次使用 time_embedded_target 作为时间嵌入。
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_target)
                # 更新粗糙样本点的位置，将前向流加到原始的粗糙样本点上。
                xyz_coarse_target=xyz_coarse_sampled + flow_fw
            
            # 检查 rays 字典是否包含键 "time_embedded_dentrg"。
            if "time_embedded_dentrg" in rays.keys():
                # 如果上述条件满足，从 rays 字典中提取 "time_embedded_dentrg" 并增加一个新的轴。
                time_embedded_dentrg = rays['time_embedded_dentrg'][:,None]
                # 再次重新评估前向流，但这次使用 time_embedded_dentrg 作为时间嵌入。
                flow_fw = evaluate_mlp(model_flowfw, xyz_coarse_embedded, 
                          chunk=chunk//N_samples, xyz=xyz_coarse_sampled,code=time_embedded_dentrg)
                # 再次更新粗糙样本点的位置，将前向流加到原始的粗糙样本点上。
                # 并更新 xyz_coarse_dentrg。
                xyz_coarse_dentrg=xyz_coarse_sampled + flow_fw
    # 这部分代码使用线性混合皮肤算法（LBS）和高斯 MLP 皮肤算法来变形点。

    # 如果模型字典 models 中包含键 bones，表明我们将要使用基于骨骼的变形方法。
    elif 'bones' in models.keys():
        
        # 获取 bones_rst，这可能是骨骼的静态（或者说是初始/休息）姿态信息。
        bones_rst = models['bones_rst']
        # 从 rays 字典中获取 bone_rts，这是与射线相关的骨骼的旋转和平移（RTS）信息，用于前向变形。
        bone_rts_fw = rays['bone_rts']
        # 获取 skin_aux，这可能是用于变形的辅助信息，例如皮肤权重或其他与皮肤相关的参数。
        skin_aux = models['skin_aux']
        # 获取 rest_pose_code，这是表示静态姿态的编码信息。
        rest_pose_code =  models['rest_pose_code']
        # 调用 rest_pose_code 函数，输入是一个零张量，转换为长整型并移到 bones_rst 所在的设备（例如CPU或GPU），这可能是用来获取骨骼静态姿态的特定编码。
        rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(bones_rst.device))
        
        # 检查 models 字典是否包含 nerf_skin。如果包含，则获取它；
        # 如果不包含，则将 nerf_skin 设置为 None。nerf_skin 可能是与NeRF模型关联的皮肤权重信息。
        if 'nerf_skin' in models.keys():
            # compute delta skinning weights of bs, N, B
            nerf_skin = models['nerf_skin'] 
        else:
            nerf_skin = None
        # 从 rays 字典中获取时间嵌入信息，并通过添加一个新轴来转换为二维数组。
        time_embedded = rays['time_embedded'][:,None]
        # coords after deform
        # 执行骨骼变换函数 bone_transform，将静态姿态 bones_rst 和旋转平移信息 bone_rts_fw 结合起来，计算出变形后的骨骼位置。参数 is_vec=True 表明输入是向量形式。
        bones_dfm = bone_transform(bones_rst, bone_rts_fw, is_vec=True)
        # 调用 gauss_mlp_skinning 函数进行高斯MLP皮肤算法变形，输入是采样的点 xyz_coarse_sampled，空间嵌入函数 embedding_xyz，
        # 变形后的骨骼 bones_dfm，时间嵌入 time_embedded，以及可能的皮肤权重 nerf_skin 和辅助信息 skin_aux。
        skin_backward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                    bones_dfm, time_embedded,  nerf_skin, skin_aux=skin_aux)

        # backward skinning
        # 执行线性混合皮肤算法 lbs，将原始骨骼 bones_rst 和变形信息 bone_rts_fw 应用于通过高斯MLP得到的皮肤权重 skin_backward，
        # 以及采样的点 xyz_coarse_sampled，得到最终变形后的点。
        xyz_coarse_sampled, bones_dfm = lbs(bones_rst, 
                                                  bone_rts_fw, 
                                                  skin_backward,
                                                  xyz_coarse_sampled,
                                                  )
        # 如果 fine_iter 为真，这表示需要进行一些额外的计算，如循环一致性和刚度损失。
        if fine_iter:
            #if opts.dist_corresp:
            # 同样使用高斯MLP皮肤算法计算向前变形，但是使用的是静态姿态信息 rest_pose_code。
            skin_forward = gauss_mlp_skinning(xyz_coarse_sampled, embedding_xyz, 
                        bones_rst,rest_pose_code,  nerf_skin, skin_aux=skin_aux)

            # 使用 lbs 函数计算循环变形，但是这次是从静态姿态变形到当前姿态，backward=False 表示这是一个前向变形。
            # cycle loss (in the joint canonical space)
            xyz_coarse_frame_cyc,_ = lbs(bones_rst, bone_rts_fw,
                              skin_forward, xyz_coarse_sampled, backward=False)

            # 计算原始帧 xyz_coarse_frame 和循环后的帧 xyz_coarse_frame_cyc 之间的差异的L2范数，作为循环一致性损失。
            frame_cyc_dis = (xyz_coarse_frame - xyz_coarse_frame_cyc).norm(2,-1)
            
            # rigidity loss (not used as optimization objective)
            # 获取骨骼的数量。
            num_bone = bones_rst.shape[0] 
            # 将 bone_rts_fw 重塑为一个三维数组，第二维是骨骼的数量，第三维是12，这可能代表骨骼的旋转和平移参数。
            bone_fw_reshape = bone_rts_fw.view(-1,num_bone,12)
            # 获取每个骨骼的平移参数。
            bone_trn = bone_fw_reshape[:,:,9:12]
            # 获取每个骨骼的旋转参数，并将其变形为3x3的矩阵。
            bone_rot = bone_fw_reshape[:,:,0:9].view(-1,num_bone,3,3)
            # 计算骨骼的刚度损失，包括平移参数的平方和以及旋转角度的计算。
            frame_rigloss = bone_trn.pow(2).sum(-1)+rot_angle(bone_rot)
            
            # 如果选项 dist_corresp 为真，并且 rays 字典中有 bone_rts_target 键，执行以下代码。
            if opts.dist_corresp and 'bone_rts_target' in rays.keys():
                # 获取目标骨骼的变形信息。
                bone_rts_target = rays['bone_rts_target']
                # 使用 lbs 函数计算从静态姿态到目标姿态的变形。
                xyz_coarse_target,_ = lbs(bones_rst, bone_rts_target, 
                                   skin_forward, xyz_coarse_sampled,backward=False)
            # 如果选项 dist_corresp 为真，并且 rays 字典中有 bone_rts_dentrg 键，执行以下代码。
            if opts.dist_corresp and 'bone_rts_dentrg' in rays.keys():
                # 获取另一组目标骨骼的变形信息。
                bone_rts_dentrg = rays['bone_rts_dentrg']
                # 再次使用 lbs 函数计算从静态姿态到这组目标姿态的变形。
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

# 这个函数在实现论文里的公式2，使用mlp去预测密度。
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


