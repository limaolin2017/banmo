# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import time
import pdb
import numpy as np
from absl import flags
import cv2
import time

import mcubes
from nnutils import banmo
import subprocess
from torch.utils.tensorboard import SummaryWriter
from kmeans_pytorch import kmeans
import torch.distributed as dist
import torch.nn.functional as F
import trimesh
import torchvision
from torch.autograd import Variable
from collections import defaultdict
from pytorch3d import transforms
from torch.nn.utils import clip_grad_norm_
from matplotlib.pyplot import cm

from nnutils.geom_utils import lbs, reinit_bones, warp_bw, warp_fw, vec_to_sim3,\
                               obj_to_cam, get_near_far, near_far_to_bound, \
                               compute_point_visibility, process_so3_seq, \
                               ood_check_cse, align_sfm_sim3, gauss_mlp_skinning, \
                               correct_bones
from nnutils.nerf import grab_xyz_weights
from ext_utils.flowlib import flow_to_image
from utils.io import mkdir_p
from nnutils.vis_utils import image_grid
from dataloader import frameloader
from utils.io import save_vid, draw_cams, extract_data_info, merge_dict,\
        render_root_txt, save_bones, draw_cams_pair, get_vertex_colors
from utils.colors import label_colormap

class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """
    for multi-gpu access
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)
    
class v2s_trainer():
    def __init__(self, opts, is_eval=False):
        self.opts = opts
        self.is_eval=is_eval
        self.local_rank = opts.local_rank
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.logname)
        
        self.accu_steps = opts.accu_steps
        
        # write logs
        if opts.local_rank==0:
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            log_file = os.path.join(self.save_dir, 'opts.log')
            if not self.is_eval:
                if os.path.exists(log_file):
                    os.remove(log_file)
                opts.append_flags_into_file(log_file)

    def define_model(self, data_info):
        opts = self.opts
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = banmo.banmo(opts, data_info)
        self.model.forward = self.model.forward_default
        self.num_epochs = opts.num_epochs

        # load model
        if opts.model_path!='':
            self.load_network(opts.model_path, is_eval=self.is_eval)

        if self.is_eval:
            self.model = self.model.to(self.device)
        else:
            # ddp
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = self.model.to(self.device)

            self.model = DataParallelPassthrough(
                    self.model,
                    device_ids=[opts.local_rank],
                    output_device=opts.local_rank,
                    find_unused_parameters=True,
            )
        return
    
    def init_dataset(self):
        opts = self.opts
        opts_dict = {}
        opts_dict['n_data_workers'] = opts.n_data_workers
        opts_dict['batch_size'] = opts.batch_size
        opts_dict['seqname'] = opts.seqname
        opts_dict['img_size'] = opts.img_size
        opts_dict['ngpu'] = opts.ngpu
        opts_dict['local_rank'] = opts.local_rank
        opts_dict['rtk_path'] = opts.rtk_path
        opts_dict['preload']= False
        opts_dict['accu_steps'] = opts.accu_steps

        if self.is_eval and opts.rtk_path=='' and opts.model_path!='':
            # automatically load cameras in the logdir
            model_dir = opts.model_path.rsplit('/',1)[0]
            cam_dir = '%s/init-cam/'%model_dir
            if os.path.isdir(cam_dir):
                opts_dict['rtk_path'] = cam_dir

        self.dataloader = frameloader.data_loader(opts_dict)
        if opts.lineload:
            opts_dict['lineload'] = True
            opts_dict['multiply'] = True # multiple samples in dataset
            self.trainloader = frameloader.data_loader(opts_dict)
            opts_dict['lineload'] = False
            del opts_dict['multiply']
        else:
            opts_dict['multiply'] = True
            self.trainloader = frameloader.data_loader(opts_dict)
            del opts_dict['multiply']
        opts_dict['img_size'] = opts.render_size
        self.evalloader = frameloader.eval_loader(opts_dict)

        # compute data offset
        data_info = extract_data_info(self.evalloader)
        return data_info
    
    def init_training(self):
        opts = self.opts
        # set as module attributes since they do not change across gpus
        self.model.module.final_steps = self.num_epochs * \
                                min(200,len(self.trainloader)) * opts.accu_steps
        # ideally should be greater than 200 batches

        params_nerf_coarse=[]
        params_nerf_beta=[]
        params_nerf_feat=[]
        params_nerf_beta_feat=[]
        params_nerf_fine=[]
        params_nerf_unc=[]
        params_nerf_flowbw=[]
        params_nerf_skin=[]
        params_nerf_vis=[]
        params_nerf_root_rts=[]
        params_nerf_body_rts=[]
        params_root_code=[]
        params_pose_code=[]
        params_env_code=[]
        params_vid_code=[]
        params_bones=[]
        params_skin_aux=[]
        params_ks=[]
        params_nerf_dp=[]
        params_csenet=[]
        for name,p in self.model.named_parameters():
            if 'nerf_coarse' in name and 'beta' not in name:
                params_nerf_coarse.append(p)
            elif 'nerf_coarse' in name and 'beta' in name:
                params_nerf_beta.append(p)
            elif 'nerf_feat' in name and 'beta' not in name:
                params_nerf_feat.append(p)
            elif 'nerf_feat' in name and 'beta' in name:
                params_nerf_beta_feat.append(p)
            elif 'nerf_fine' in name:
                params_nerf_fine.append(p)
            elif 'nerf_unc' in name:
                params_nerf_unc.append(p)
            elif 'nerf_flowbw' in name or 'nerf_flowfw' in name:
                params_nerf_flowbw.append(p)
            elif 'nerf_skin' in name:
                params_nerf_skin.append(p)
            elif 'nerf_vis' in name:
                params_nerf_vis.append(p)
            elif 'nerf_root_rts' in name:
                params_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                params_nerf_body_rts.append(p)
            elif 'root_code' in name:
                params_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                params_pose_code.append(p)
            elif 'env_code' in name:
                params_env_code.append(p)
            elif 'vid_code' in name:
                params_vid_code.append(p)
            elif 'module.bones' == name:
                params_bones.append(p)
            elif 'module.skin_aux' == name:
                params_skin_aux.append(p)
            elif 'module.ks_param' == name:
                params_ks.append(p)
            elif 'nerf_dp' in name:
                params_nerf_dp.append(p)
            elif 'csenet' in name:
                params_csenet.append(p)
            else: continue
            if opts.local_rank==0:
                print('optimized params: %s'%name)

        self.optimizer = torch.optim.AdamW(
            [{'params': params_nerf_coarse},
             {'params': params_nerf_beta},
             {'params': params_nerf_feat},
             {'params': params_nerf_beta_feat},
             {'params': params_nerf_fine},
             {'params': params_nerf_unc},
             {'params': params_nerf_flowbw},
             {'params': params_nerf_skin},
             {'params': params_nerf_vis},
             {'params': params_nerf_root_rts},
             {'params': params_nerf_body_rts},
             {'params': params_root_code},
             {'params': params_pose_code},
             {'params': params_env_code},
             {'params': params_vid_code},
             {'params': params_bones},
             {'params': params_skin_aux},
             {'params': params_ks},
             {'params': params_nerf_dp},
             {'params': params_csenet},
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)

        if self.model.root_basis=='exp':
            lr_nerf_root_rts = 10
        elif self.model.root_basis=='cnn':
            lr_nerf_root_rts = 0.2
        elif self.model.root_basis=='mlp':
            lr_nerf_root_rts = 1 
        elif self.model.root_basis=='expmlp':
            lr_nerf_root_rts = 1 
        else: print('error'); exit()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
                        [opts.learning_rate, # params_nerf_coarse
                         opts.learning_rate, # params_nerf_beta
                         opts.learning_rate, # params_nerf_feat
                      10*opts.learning_rate, # params_nerf_beta_feat
                         opts.learning_rate, # params_nerf_fine
                         opts.learning_rate, # params_nerf_unc
                         opts.learning_rate, # params_nerf_flowbw
                         opts.learning_rate, # params_nerf_skin
                         opts.learning_rate, # params_nerf_vis
        lr_nerf_root_rts*opts.learning_rate, # params_nerf_root_rts
                         opts.learning_rate, # params_nerf_body_rts
        lr_nerf_root_rts*opts.learning_rate, # params_root_code
                         opts.learning_rate, # params_pose_code
                         opts.learning_rate, # params_env_code
                         opts.learning_rate, # params_vid_code
                         opts.learning_rate, # params_bones
                      10*opts.learning_rate, # params_skin_aux
                      10*opts.learning_rate, # params_ks
                         opts.learning_rate, # params_nerf_dp
                         opts.learning_rate, # params_csenet
            ],
            int(self.model.module.final_steps/self.accu_steps),
            pct_start=2./self.num_epochs, # use 2 epochs to warm up
            cycle_momentum=False, 
            anneal_strategy='linear',
            final_div_factor=1./5, div_factor = 25,
            )
    
    def save_network(self, epoch_label, prefix=''):
        if self.opts.local_rank==0:
            param_path = '%s/%sparams_%s.pth'%(self.save_dir,prefix,epoch_label)
            save_dict = self.model.state_dict()
            torch.save(save_dict, param_path)

            var_path = '%s/%svars_%s.npy'%(self.save_dir,prefix,epoch_label)
            latest_vars = self.model.latest_vars.copy()
            del latest_vars['fp_err']  
            del latest_vars['flo_err']   
            del latest_vars['sil_err'] 
            del latest_vars['flo_err_hist']
            np.save(var_path, latest_vars)
            return
    
    @staticmethod
    def rm_module_prefix(states, prefix='module'):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[:len(prefix)] == prefix:
                i = i[len(prefix)+1:]
            new_dict[i] = v
        return new_dict

    def load_network(self,model_path=None, is_eval=True, rm_prefix=True):
        opts = self.opts
        states = torch.load(model_path,map_location='cpu')
        if rm_prefix: states = self.rm_module_prefix(states)
        var_path = model_path.replace('params', 'vars').replace('.pth', '.npy')
        latest_vars = np.load(var_path,allow_pickle=True)[()]
        
        if is_eval:
            # load variables
            self.model.latest_vars = latest_vars
        
        # if size mismatch, delete all related variables
        if rm_prefix and states['near_far'].shape[0] != self.model.near_far.shape[0]:
            print('!!!deleting video specific dicts due to size mismatch!!!')
            self.del_key( states, 'near_far') 
            self.del_key( states, 'root_code.weight') # only applies to root_basis=mlp
            self.del_key( states, 'pose_code.weight')
            self.del_key( states, 'pose_code.basis_mlp.weight')
            self.del_key( states, 'nerf_body_rts.0.weight')
            self.del_key( states, 'nerf_body_rts.0.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.0.weight')
            self.del_key( states, 'nerf_root_rts.root_code.weight')
            self.del_key( states, 'nerf_root_rts.root_code.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.delta_rt.0.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.base_rt.se3')
            self.del_key( states, 'nerf_root_rts.delta_rt.0.weight')
            self.del_key( states, 'env_code.weight')
            self.del_key( states, 'env_code.basis_mlp.weight')
            if 'vid_code.weight' in states.keys():
                self.del_key( states, 'vid_code.weight')
            if 'ks_param' in states.keys():
                self.del_key( states, 'ks_param')

            # delete pose basis(backbones)
            if not opts.keep_pose_basis:
                del_key_list = []
                for k in states.keys():
                    if 'nerf_body_rts' in k or 'nerf_root_rts' in k:
                        del_key_list.append(k)
                for k in del_key_list:
                    print(k)
                    self.del_key( states, k)
    
        if rm_prefix and opts.lbs and states['bones'].shape[0] != self.model.bones.shape[0]:
            self.del_key(states, 'bones')
            states = self.rm_module_prefix(states, prefix='nerf_skin')
            states = self.rm_module_prefix(states, prefix='nerf_body_rts')


        # load some variables
        # this is important for volume matching
        if latest_vars['obj_bound'].size==1:
            latest_vars['obj_bound'] = latest_vars['obj_bound'] * np.ones(3)
        self.model.latest_vars['obj_bound'] = latest_vars['obj_bound'] 

        # load nerf_coarse, nerf_bone/root (not code), nerf_vis, nerf_feat, nerf_unc
        #TODO somehow, this will reset the batch stats for 
        # a pretrained cse model, to keep those, we want to manually copy to states
        if opts.ft_cse and \
          'csenet.net.backbone.fpn_lateral2.weight' not in states.keys():
            self.add_cse_to_states(self.model, states)
        self.model.load_state_dict(states, strict=False)

        return

    @staticmethod 
    def add_cse_to_states(model, states):
        states_init = model.state_dict()
        for k in states_init.keys():
            v = states_init[k]
            if 'csenet' in k:
                states[k] = v

    def eval_cam(self, idx_render=None): 
        """
        idx_render: list of frame index to render
        """
        opts = self.opts
        with torch.no_grad():
            self.model.eval()
            # load data
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = False
            batch = []
            for i in idx_render:
                batch.append( self.evalloader.dataset[i] )
            batch = self.evalloader.collate_fn(batch)
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = True

            #这里好像是加速模型运行的速度
            #TODO can be further accelerated
            self.model.convert_batch_input(batch)

            if opts.unc_filter:
                # process densepoe feature
                valid_list, error_list = ood_check_cse(self.model.dp_feats, 
                                        self.model.dp_embed, 
                                        self.model.dps.long())
                valid_list = valid_list.cpu().numpy()
                error_list = error_list.cpu().numpy()
            else:
                valid_list = np.ones( len(idx_render))
                error_list = np.zeros(len(idx_render))

            self.model.convert_root_pose()
            rtk = self.model.rtk
            kaug = self.model.kaug

            #TODO may need to recompute after removing the invalid predictions
            # need to keep this to compute near-far planes
            self.model.save_latest_vars()
                
            # extract mesh sequences
            aux_seq = {
                       'is_valid':[],
                       'err_valid':[],
                       'rtk':[],
                       'kaug':[],
                       'impath':[],
                       'masks':[],
                       }
            for idx,_ in enumerate(idx_render):
                frameid=self.model.frameid[idx]
                #这里出现的.cpu().numpy()的含义是将数据转移到cpu然后将数据变成numpy数组
                    if opts.local_rank==0: 
                    print('extracting frame %d'%(frameid.cpu().numpy()))
                aux_seq['rtk'].append(rtk[idx].cpu().numpy())
                aux_seq['kaug'].append(kaug[idx].cpu().numpy())
                aux_seq['masks'].append(self.model.masks[idx].cpu().numpy())
                aux_seq['is_valid'].append(valid_list[idx])
                aux_seq['err_valid'].append(error_list[idx])
                
                impath = self.model.impath[frameid.long()]
                aux_seq['impath'].append(impath)
        return aux_seq
  
    def eval(self, idx_render=None, dynamic_mesh=False): 
        """
        idx_render: list of frame index to render
        dynamic_mesh: whether to extract canonical shape, or dynamic shape
        """
        opts = self.opts
        with torch.no_grad():
            self.model.eval()

            # run marching cubes on canonical shape
            mesh_dict_rest = self.extract_mesh(self.model, opts.chunk, \
                                         opts.sample_grid3d, opts.mc_threshold)

            # choose a grid image or the whold video
            if idx_render is None: # render 9 frames
                idx_render = np.linspace(0,len(self.evalloader)-1, 9, dtype=int)

            # render
            chunk=opts.rnd_frame_chunk
            rendered_seq = defaultdict(list)
            aux_seq = {'mesh_rest': mesh_dict_rest['mesh'],
                       'mesh':[],
                       'rtk':[],
                       'impath':[],
                       'bone':[],}
            for j in range(0, len(idx_render), chunk):
                batch = []
                idx_chunk = idx_render[j:j+chunk]
                for i in idx_chunk:
                    batch.append( self.evalloader.dataset[i] )
                batch = self.evalloader.collate_fn(batch)
                rendered = self.render_vid(self.model, batch) 
            
                for k, v in rendered.items():
                    rendered_seq[k] += [v]
                    
                hbs=len(idx_chunk)
                sil_rszd = F.interpolate(self.model.masks[:hbs,None], 
                            (opts.render_size, opts.render_size))[:,0,...,None]
                rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)[:hbs]]
                rendered_seq['sil'] += [self.model.masks[...,None]      [:hbs]]
                rendered_seq['flo'] += [self.model.flow.permute(0,2,3,1)[:hbs]]
                rendered_seq['dpc'] += [self.model.dp_vis[self.model.dps.long()][:hbs]]
                rendered_seq['occ'] += [self.model.occ[...,None]      [:hbs]]
                rendered_seq['feat']+= [self.model.dp_feats.std(1)[...,None][:hbs]]
                rendered_seq['flo_coarse'][-1]       *= sil_rszd 
                rendered_seq['img_loss_samp'][-1]    *= sil_rszd 
                if 'frame_cyc_dis' in rendered_seq.keys() and \
                    len(rendered_seq['frame_cyc_dis'])>0:
                    rendered_seq['frame_cyc_dis'][-1] *= 255/rendered_seq['frame_cyc_dis'][-1].max()
                    rendered_seq['frame_rigloss'][-1] *= 255/rendered_seq['frame_rigloss'][-1].max()
                if opts.use_embed:
                    rendered_seq['pts_pred'][-1] *= sil_rszd 
                    rendered_seq['pts_exp'] [-1] *= rendered_seq['sil_coarse'][-1]
                    rendered_seq['feat_err'][-1] *= sil_rszd
                    rendered_seq['feat_err'][-1] *= 255/rendered_seq['feat_err'][-1].max()
                if opts.use_proj:
                    rendered_seq['proj_err'][-1] *= sil_rszd
                    rendered_seq['proj_err'][-1] *= 255/rendered_seq['proj_err'][-1].max()
                if opts.use_unc:
                    rendered_seq['unc_pred'][-1] -= rendered_seq['unc_pred'][-1].min()
                    rendered_seq['unc_pred'][-1] *= 255/rendered_seq['unc_pred'][-1].max()

                # extract mesh sequences
                for idx in range(len(idx_chunk)):
                    frameid=self.model.frameid[idx].long()
                    embedid=self.model.embedid[idx].long()
                    print('extracting frame %d'%(frameid.cpu().numpy()))
                    # run marching cubes
                    if dynamic_mesh:
                        if not opts.queryfw:
                           mesh_dict_rest=None 
                        mesh_dict = self.extract_mesh(self.model,opts.chunk,
                                            opts.sample_grid3d, opts.mc_threshold,
                                        embedid=embedid, mesh_dict_in=mesh_dict_rest)
                        mesh=mesh_dict['mesh']
                        if mesh_dict_rest is not None and opts.ce_color:
                            mesh.visual.vertex_colors = mesh_dict_rest['mesh'].\
                                   visual.vertex_colors # assign rest surface color
                        else:
                            # get view direction 
                            obj_center = self.model.rtk[idx][:3,3:4]
                            cam_center = -self.model.rtk[idx][:3,:3].T.matmul(obj_center)[:,0]
                            view_dir = torch.cuda.FloatTensor(mesh.vertices, device=self.device) \
                                            - cam_center[None]
                            vis = get_vertex_colors(self.model, mesh_dict_rest['mesh'], 
                                                    frame_idx=idx, view_dir=view_dir)
                            mesh.visual.vertex_colors[:,:3] = vis*255

                        # save bones
                        if 'bones' in mesh_dict.keys():
                            bone = mesh_dict['bones'][0].cpu().numpy()
                            aux_seq['bone'].append(bone)
                    else:
                        mesh=mesh_dict_rest['mesh']
                    aux_seq['mesh'].append(mesh)

                    # save cams
                    aux_seq['rtk'].append(self.model.rtk[idx].cpu().numpy())
                    
                    # save image list
                    impath = self.model.impath[frameid]
                    aux_seq['impath'].append(impath)

            # save canonical mesh and extract skinning weights
            mesh_rest = aux_seq['mesh_rest']
            if len(mesh_rest.vertices)>100:
                self.model.latest_vars['mesh_rest'] = mesh_rest
            if opts.lbs:
                bones_rst = self.model.bones
                bones_rst,_ = correct_bones(self.model, bones_rst)
                # compute skinning color
                if mesh_rest.vertices.shape[0]>100:
                    rest_verts = torch.Tensor(mesh_rest.vertices).to(self.device)
                    nerf_skin = self.model.nerf_skin if opts.nerf_skin else None
                    rest_pose_code = self.model.rest_pose_code(torch.Tensor([0])\
                                            .long().to(self.device))
                    skins = gauss_mlp_skinning(rest_verts[None], 
                            self.model.embedding_xyz,
                            bones_rst, rest_pose_code, 
                            nerf_skin, skin_aux=self.model.skin_aux)[0]
                    skins = skins.cpu().numpy()
   
                    num_bones = skins.shape[-1]
                    colormap = label_colormap()
                    # TODO use a larger color map
                    colormap = np.repeat(colormap[None],4,axis=0).reshape(-1,3)
                    colormap = colormap[:num_bones]
                    colormap = (colormap[None] * skins[...,None]).sum(1)

                    mesh_rest_skin = mesh_rest.copy()
                    mesh_rest_skin.visual.vertex_colors = colormap
                    aux_seq['mesh_rest_skin'] = mesh_rest_skin

                aux_seq['bone_rest'] = bones_rst.cpu().numpy()
        
            # draw camera trajectory
            suffix_id=0
            if hasattr(self.model, 'epoch'):
                suffix_id = self.model.epoch
            if opts.local_rank==0:
                mesh_cam = draw_cams(aux_seq['rtk'])
                mesh_cam.export('%s/mesh_cam-%02d.obj'%(self.save_dir,suffix_id))
            
                mesh_path = '%s/mesh_rest-%02d.obj'%(self.save_dir,suffix_id)
                mesh_rest.export(mesh_path)

                if opts.lbs:
                    bone_rest = aux_seq['bone_rest']
                    bone_path = '%s/bone_rest-%02d.obj'%(self.save_dir,suffix_id)
                    save_bones(bone_rest, 0.1, bone_path)

            # save images
            for k,v in rendered_seq.items():
                rendered_seq[k] = torch.cat(rendered_seq[k],0)
                ##TODO
                #if opts.local_rank==0:
                #    print('saving %s to gif'%k)
                #    is_flow = self.isflow(k)
                #    upsample_frame = min(30,len(rendered_seq[k]))
                #    save_vid('%s/%s'%(self.save_dir,k), 
                #            rendered_seq[k].cpu().numpy(), 
                #            suffix='.gif', upsample_frame=upsample_frame, 
                #            is_flow=is_flow)

        return rendered_seq, aux_seq

    def train(self):
        opts = self.opt
        '''
        如果是多gpus训练，local_rank的数值会来回切换吗？

        在多GPU训练的情况下，每个GPU进程通常会在训练开始时被分配一个固定的 `local_rank` 值，
        这个值在训练过程中是不会变化的。每个GPU进程的 `local_rank` 是唯一的，并且用于区分不同的GPU。
        例如，在一个有四个GPU的系统上进行分布式训练时，可能会有以下的 `local_rank` 分配：
        - GPU 0 -> `local_rank = 0`
        - GPU 1 -> `local_rank = 1`
        - GPU 2 -> `local_rank = 2`
        - GPU 3 -> `local_rank = 3`

        在整个训练过程中，每个GPU进程会保持它被分配的 `local_rank`。
        在一些分布式训练框架中，
        例如PyTorch的 `torch.distributed`，`local_rank` 是在训练脚本启动时通过环境变量或命令行参数传递给每个进程的。
        这个值会被用来设置进程应该使用的GPU，以及在需要进行跨进程通信时确定通信的对象。
        '''

        '''
        在分布式训练中，local_rank 主要用于在单个节点上区分进程和分配 GPU。
        而 rank（或称为 global_rank）则用于在所有进程之间进行区分。
        例如，如果你有 2 个节点，每个节点有 2 个 GPU，那么可能的 rank 和 local_rank 分配如下：

        节点 1
        GPU 0 -> local_rank = 0, global_rank = 0
        GPU 1 -> local_rank = 1, global_rank = 1
        节点 2
        GPU 0 -> local_rank = 0, global_rank = 2
        GPU 1 -> local_rank = 1, global_rank = 3
        '''


        '''
        opts.local_rank == 1 意味着什么？
        这意味着当前的代码正在第二个GPU上执行。这通常用于决定是否执行某些操作。
        例如，在代码的日志记录部分，可能只想在 local_rank = 0 的GPU上执行，
        以避免多个进程写入同一个日志文件。
        '''

        '''
        'opts.local_rank==1'意味着什么？
        这意味着当前进程在其节点上的GPU排序是第二位。
        这个进程可能负责执行计算任务，但通常不负责执行日志记录或保存模型等任务。

        'opts.local_rank==0'意味着程序在主线程上运行，一般来说我们会记录日志，对吗？
        是的，opts.local_rank==0 通常用于确定哪个进程会负责输出日志或进行与文件系统交互的操作，如保存模型。
        在多节点多GPU的设置中，每个节点上的 local_rank == 0 的进程通常负责这类“主进程”任务，
        但是只有全局排名为0的进程（即第一个节点上的 local_rank == 0 的进程）才是整个分布式训练中的主进程。       
        
        '''

        if opts.local_rank==0:
            log = SummaryWriter('%s/%s'%(opts.checkpoint_dir,opts.logname), comment=opts.logname)
        else: log=None

        #重置模型的进度为0
        '''
        total_steps 是一个累积变量，用于追踪模型在整个训练过程中处理了多少批次（batches）或步骤（steps）。
        将其设置为0通常表示训练过程从头开始，或者开始一个新的阶段，在这个新阶段中，之前的步骤计数不再被考虑。
        '''
        self.model.module.total_steps = 0
        '''
        progress 是一个通常用于表示训练进度的变量。它可能是一个介于0和1之间的数字，表示从训练开始到结束的完成比例。
        设置为0表示训练刚刚开始，或者是在开始新的训练阶段时的初始状态。
        '''
        self.model.module.progress = 0

        #又一次设置随机种子
        '''torch.manual_seed(8) 设置了CPU随机数生成器的种子。
        这意味着所有依赖于CPU生成的随机数的操作（例如随机初始化参数、随机打乱数据等）将会从这个种子开始产生随机数。
        

        为什么要设置两遍随机数？

        torch.manual_seed(8) 设置了CPU的随机数生成器的种子。
        torch.cuda.manual_seed(1) 设置了当前CUDA设备的随机数生成器的种子。
        这样做是因为PyTorch为CPU和GPU分别维护了不同的随机数生成器。
        如果你使用的是GPU加速的计算，那么设置CUDA的种子是必要的，以确保在GPU上进行的操作（如权重初始化）也是可重复的。

        
        '''
        torch.manual_seed(8)  # do it again
        '''
        torch.cuda.manual_seed(1) 设置了当前CUDA设备的随机数生成器的种子。
        这对所有依赖于GPU生成的随机数的操作产生影响，例如在GPU上进行的随机初始化操作。

        随机数设置为8意味着什么？
        随机数种子设置为8并没有特殊的含义，它只是一个任意选择的数字。
        种子的具体值并不重要，重要的是使用相同的种子可以生成相同的随机数序列。
        
        随机数设置为1意味着什么？
        和CPU的种子一样，CUDA种子设置为1也是为了确定性。
        它也只是一个任意选择的数字，用来确保每次在GPU上运行时都得到相同的结果。
        '''
        torch.cuda.manual_seed(1)

        # disable bones before warmup epochs are finished
        # 检查是否启用了线性混合蒙皮（Linear Blend Skinning，LBS）选项。
        '''
        
        线性混合蒙皮是一种计算方法，
        在3D角色动画中用于使皮肤或模型的网格随着内部骨骼结构而移动和变形。
        当这些骨骼移动时，顶点的新位置是通过这些骨骼的变换和权重的线性组合来计算的。
        这种方法假设每个网格顶点都受到一个或多个“骨骼”影响。
        每个骨骼对顶点的影响通过权重来衡量。
        这些权重反映了顶点随着骨骼运动的程度。
        当骨骼移动时，它会根据这些权重来混合多个骨骼对每个顶点的影响，
        以此来计算最终的顶点位置。

        权重表达骨骼影响的例子：
        假设我们有一个简单的模型，比如一个手臂，手臂有两根骨骼：上臂骨和前臂骨。
        如果有一个顶点位于肘部，那么这个顶点可能有以下权重：
        上臂骨骼权重：0.5
        前臂骨骼权重：0.5
        这意味着如果上臂骨骼旋转，这个顶点会受到一半的影响，
        如果前臂骨骼旋转，顶点也会受到一半的影响。
        在实际的应用中，权重会更加复杂，并且是根据实际模型的几何形状和预期的动画效果来细致调整的。

        在LBS中，如果一个顶点相对于两根骨骼都有0.5的权重，那么这个顶点在变形时会同时考虑两根骨骼的变换。
        具体来说，如果上臂骨骼旋转，顶点会根据上臂骨骼的旋转变换移动一半的相应距离或角度；如果前臂骨骼旋转，
        顶点也会根据前臂骨骼的变换移动另一半的距离或角度。
        这样，顶点的最终位置是两个骨骼变换的平均效果，从而在骨骼连接的区域（如肘部）创建一个平滑的过渡效果。
        


        代码中删除'nerf_skin'可能意味着在这个训练阶段不想使用NeRF来处理皮肤的细节。
        这可能是为了简化问题，让模型集中在更基础的结构学习上，或者是为了避免在这个阶段处理过于复杂的特征。

        网格（Mesh）在3D建模和图形学中通常指的是由
        顶点（vertices）、边（edges）和面（faces）组成的多边形集合，它定义了一个3D物体的表面。

        网格顶点（Mesh Vertices）是构成3D网格的点，每个顶点具有一个3D空间中的位置坐标。
        网格的形状和结构通过这些顶点及它们之间的连接关系定义。


        nerf_models 如果是一个字典，在NeRF模型的上下文中，它通常会存储各种子模块或组件的引用。
        这些子模块可能负责不同的任务，比如处理不同的数据流（如骨骼动态、皮肤特征），
        或者实现模型的不同部分（如光线投射、体积渲染等）。
        每个键通常对应一个特定的功能模块或参数集，方便在模型训练或推断时进行调用或修改。


        LBS是否是NeRF的组件之一：在标准的NeRF实现中，LBS不是内建的一部分，因为NeRF主要关注的是静态场景的渲染。
        但在扩展的NeRF模型中，特别是用于动态场景或者需要模拟物体变形的情况下，
        LBS可以作为一个组件被集成到NeRF中以实现这些功能。

        如果骨骼的渲染不是基础的、重要的，那哪些部分是呢？

        在NeRF模型的训练中，基础和重要的部分可能包括学习场景的基本3D结构、光照和颜色分布等。
        这些元素构成了场景的基础，不受特定动作或变形的影响。

        skin会依赖bones来进行训练吗?
        如果nerf_skin是指一个由bones参数控制的可变形的网络，那么它确实可能依赖于bones。
        所以这里会先禁用bone然后是nerf_skin

        '''
        if opts.lbs: 
            # 将模型中使用的骨骼数量设置为0，这可能意味着在当前训练阶段，模型不使用任何骨骼信息。
            self.model.num_bone_used = 0
            del self.model.module.nerf_models['bones']
            # 检查是否启用了lbs和nerf_skin
        if opts.lbs and opts.nerf_skin:
            del self.model.module.nerf_models['nerf_skin']

        # warmup shape
        '''
        在机器学习和特别是深度学习中，“预热”通常指的是训练过程的一部分，
        其中开始时学习率较低或其他参数被设定为帮助模型慢慢调整到较复杂任务的最优点。
        '''
        if opts.warmup_shape_ep>0:
        '''
        self.warmup_shape(log) 是调用当前类的 warmup_shape 方法，log 参数很可能是用于记录训练进度和结果的对象，
        例如TensorBoard的 SummaryWriter。这个方法会负责执行形状预热的实际步骤。
        '''
            self.warmup_shape(log)

        # CNN pose warmup or  load CNN
        # CNN姿态预热可能是指在正式开始训练模型之前，先通过一个预训练的卷积神经网络（CNN）来进行姿态估计的预热。
        # 这样做的目的通常是为了初始化网络权重，使得模型在开始训练时就有一个相对较好的起点。
        # 如果任一条件为真，就会调用 self.warmup_pose 方法，进行姿态预热或加载预训练模型。
        # log 是用于记录进度的对象，而 pose_cnn_path 是CNN模型文件的路径。

        '''
        预训练的CNN模型:
        预训练的CNN模型是一个已经在相关任务上训练过的模型，其参数已经基于之前的数据进行了学习。
        在这个上下文中，预训练的CNN模型可能用于姿态估计，
        意味着它已经在估计物体或者人的姿态上训练过，能够提供一个初始的、有用的权重集合，从而加速后续的训练过程。
        '''

        '''
        opts.use_rtk_file 这一选项表明，代码可以配置为使用一个RTK文件。
        RTK文件通常包含了摄像机参数和其他的跟踪数据。
        这里的 if opts.use_rtk_file: 检查表明，如果有RTK文件提供，代码将会按照这个文件中的数据来提取摄像机参数。
        这是一个配置选项，意味着作者或用户可以选择是否提供这样的文件来辅助训练。
        '''
        '''
        opts.warmup_pose_ep 很可能是在配置模型之前设定的一个数值，它指定了姿态预热的epoch数量。
        如果这个值大于0，则表示需要进行预热过程。预热过程是在主训练循环开始之前进行的，
        通常用于使模型参数适应训练数据，可能通过使用较低的学习率或其他特定的训练策略来完成。
        接下来，代码中的 self.warmup_pose 方法被调用，传入了 log 对象用于记录训练过程，
        以及 pose_cnn_path 从 opts 对象中获取的路径，
        这个路径可能指向一个预训练的CNN模型，用于姿态估计。
        如果 opts.warmup_pose_ep 等于0并且 opts.pose_cnn_path 为空字符串，则不进行姿态预热，
        而是执行 else 分支的代码。在这个分支中，代码检查 opts.use_rtk_file 是否为真。如果为真，
        它将模型的 use_cam 属性设置为 True，执行 self.extract_cams 方法来提取相机参数，
        并在结束后恢复 use_cam 的原始设置。如果 opts.use_rtk_file 为假，则直接执行 self.extract_cams 方法。
        这个方法的作用是从数据加载器中提取相机参数，并将它们保存到模型的状态或文件中。
        
        '''
        if opts.warmup_pose_ep>0 or opts.pose_cnn_path!='':
            self.warmup_pose(log, pose_cnn_path=opts.pose_cnn_path)
        '''
        如果既不进行预热也不加载预训练模型，模型就需要从头开始训练，这时使用预定义的相机参数文件可以提供一个固定的起点，
        尤其是在3D建模和计算机视觉任务中，相机参数对于定位和理解3D场景至关重要。
        简而言之，不进行预热和不使用预训练模型意味着模型可能需要依赖其他方式，如直接从相机参数文件中获取必要的初始化信息。
        
        '''
        else:
            # save cameras to latest vars and file
            # 这部分代码首先检查是否需要根据RTK文件（一种可能包含相机参数的文件）来设置相机参数。
            if opts.use_rtk_file:
                # 临时将模型的 use_cam 设置为 True
                self.model.module.use_cam=True
                # 提取相机参数
                self.extract_cams(self.dataloader)
                '''
                重新设置 self.model.module.use_cam=opts.use_cam 的原因可能是在执行 self.extract_cams(self.dataloader) 时，
                需要临时改变模型的 use_cam 属性。在 self.extract_cams 方法中可能需要使用相机数据来执行某些操作，
                而这些操作可能只在 use_cam 为 True 时才有效。在这些操作完成后，
                作者将 use_cam 属性重置为它的原始状态（由 opts.use_cam 指定），
                以确保模型的其它部分不会受到这一临时改变的影响。
                '''
                self.model.module.use_cam=opts.use_cam
            else:
                # 如果不需要使用RTK文件，则直接提取相机参数。
                self.extract_cams(self.dataloader)

        #TODO train mlp
        # 检查是否需要对root MLP（根多层感知机）进行预热。
        if opts.warmup_rootmlp:
            # set se3 directly
            # 从模型的最新变量中取出旋转矩阵（通常称为RTK矩阵的旋转部分）并转换成PyTorch张量。
            # RTK矩阵通常包含3x3的旋转矩阵和3x1的平移向量，这里只提取了旋转矩阵部分。
            rmat = torch.Tensor(self.model.latest_vars['rtk'][:,:3,:3])

            '''
            是的，latest_vars 通常表示模型的最新状态或变量。在这个上下文中，
            latest_vars['rtk'] 很可能存储了最新的相机参数或变换矩阵，这些参数是动态更新的，
            代表模型在训练或推断过程中当前的状态。
            rtk 通常代表相机的旋转（Rotation）和平移（Translation）参数，也就是相机的位姿。
            它将 rtk 中的旋转部分（3x3旋转矩阵）赋予了变量 rmat，
            然后进一步将这个旋转矩阵转换为四元数，并赋值给模型的相应部分


            这里是将相机参数的旋转部分赋予了模型的根部姿态的旋转部分，对吗?
            代码将这个四元数赋值给模型的根部姿态表示 nerf_root_rts 中的 se3 属性的旋转部分。
            SE(3) 是指刚体变换中的旋转（用四元数表示）和平移（用向量表示）的组合。
            这里确实是在将相机参数的旋转部分（通过四元数表示）赋给模型的根部姿态的旋转部分。
            这可能是模型预热或初始化的一部分，其中使用实际相机的旋转数据来设置网络内部的参数。

            在SE(3)变换的表示中，通常前三个元素（data[:, :3]）用于表示平移部分，而接下来的四个元素（data[:, 3:7]）
            用于表示旋转部分（在四元数形式下）。所以：
            base_rt.se3.data[:,3:] 是指分配给模型的根部姿态的旋转部分。
            这个部分正在被设置为由旋转矩阵转换得到的四元数。这个操作是在初始化或预热模型时进行的，
            以便于后续训练过程中使用正确的根部姿态的旋转。


            为什么将相机参数赋予了模型的一部分？
            在计算机视觉和3D建模任务中，相机参数是关键的，因为它们描述了相机在世界坐标系中的位置和方向。
            当模型需要理解或重建3D场景时，它需要知道每个视角的相机参数。在神经渲染（NeRF）类型的模型中，
            这些参数通常用来将3D空间的点投影到2D图像空间，或者相反，从2D图像空间重建3D点的位置。
            在预热阶段，通过将真实的或预先计算好的相机旋转参数设置到模型的一部分，
            可以帮助模型快速适应数据集的几何结构。这种预热可以减少训练时间，
            有助于模型在训练初期就能更好地理解相机的视角和位置，特别是在模型的初期训练阶段，这可以提供一个更稳定的起点。
            '''


            # 将旋转矩阵转换为四元数。四元数是一种表示旋转的方式，它可以避免万向节锁问题，
            # 并且通常用于3D图形和机器人学中的旋转表示。
            quat = transforms.matrix_to_quaternion(rmat).to(self.device)
            # 将转换得到的四元数设置为神经辐射场（NeRF）模型中的根部旋转表示（nerf_root_rts）的SE(3)变换矩阵的旋转部分。
            # 在SE(3)矩阵中，旋转通常由矩阵的前三个元素表示，而平移由最后一列表示。
            self.model.module.nerf_root_rts.base_rt.se3.data[:,3:] = quat
            # 这里代码将SE(3)的旋转部分（最后三列的前三行）设置为计算出的四元数。
        # 总结来说，这段代码是在预热阶段设置模型的根部SE(3)变换的旋转部分，使用从数据中得到的旋转矩阵直接转换成四元数并赋值。
        '''
        SE(3)的含义:
            SE(3)是指三维空间的特殊欧几里得群，它包含所有保持物体尺寸和形状不变的刚体变换，即所有的旋转和平移。
            SE(3)中的每一个元素可以表示为一个4×4的变换矩阵，该矩阵由一个3×3的旋转矩阵和一个3维的平移向量组成。
        '''
        '''
        MLP预热与CNN预热的区别:
        MLP预热通常指的是对多层感知机（MLP）这种全连接神经网络进行预训练，
        可能是为了初始化权重或者对网络进行预调整，以便它能更好地处理后续的特定任务。
        CNN预热指的是对卷积神经网络（CNN）进行预训练。
        卷积网络由于其结构特点，在处理图像数据时特别有效，预热可能涉及在相关任务上预先训练网络或调整网络权重。
        两者的主要区别在于网络的结构不同，MLP是全连接网络，而CNN含有卷积层，通常用于处理具有空间层次结构的数据，如图像。

        为什么预热会有两个不同的神经网络:
        在某些复杂任务，如3D重建或姿态估计中，可能会有多个网络分别负责不同的子任务。
        例如，一个网络可能负责理解场景的结构（MLP），而另一个网络可能负责提取图像特征（CNN）。
        使用不同的预热策略可以确保每个网络都能以最佳状态开始训练，从而提高最终模型的性能。


        在NeRF模型中，MLP（多层感知器）通常用于从空间坐标和视角预测颜色和密度，进而重建场景的3D结构。

        SE(3)变换矩阵是NeRF模型的组件之一吗？ 
        是的，在NeRF模型中，SE(3)变换矩阵通常用于表示相机或物体的位置和旋转。
        这是渲染过程中必不可少的，因为它决定了从哪个角度和位置观察场景，从而影响渲染的结果。
        在某些NeRF模型的变种中，SE(3)变换也可能用于描述场景中动态物体的运动。

        “根部旋转”通常指的是在层次结构模型中，如骨骼动画，最顶层或基础的旋转。
        例如，在人体骨骼动画中，根部可能是骨盆或躯干的中心，所有其他的骨骼和关节都是相对于这个根部的。

        NeRF模型中的根部旋转表示:
        在NeRF模型中，根部旋转表示通常是一个SE3变换，
        它定义了场景的全局参照系或者一个主要物体（如人体模型的根部骨骼）的旋转。
        它是场景或物体的“根”，所有其他的部分都相对于这个根来进行变换。

        设置相机的RTK为根部旋转:
        代码并非直接将相机的rtk设置为根部旋转，而是将从相机参数中提取的旋转矩阵转换为四元数，
        然后将这个四元数设置为NeRF模型的根部旋转的表示。
        这个过程是因为NeRF模型的根部旋转需要一个稳定和准确的旋转表示，而四元数提供了这样的特性。

        '''

        # clear buffers for pytorch1.10+
        try: self.model._assign_modules_buffers()
        except: pass
        '''
        用来分配或重置模型的缓冲区（buffers）。
        缓冲区通常用于保存跨多次前向传播迭代不变的数据，比如批量归一化层（BatchNorm）中的运行时均值和方差。
        '''

        # set near-far plane
        if opts.model_path=='':
            self.reset_nf()

        # reset idk in latest_vars
        self.model.module.latest_vars['idk'][:] = 0.
   
        #保存已经加载的与姿态相关的权重
        #TODO save loaded wts of posecs
        # 这段代码的目的是在某些训练阶段冻结（freeze）神经辐射场（NeRF）的某些权重。在训练深度学习模型时，
        # 有时我们希望固定（或“冻结”）某些层的权重，以便它们在后续的训练迭代中不会更新。
        # 这样做可以避免已经学习到有用特征的权重被破坏，尤其是在微调（fine-tuning）阶段或者当我们想要固定某些特征表示时。
        if opts.freeze_coarse:
            self.model.module.shape_xyz_wt = \
                grab_xyz_weights(self.model.module.nerf_coarse, clone=True)
            self.model.module.skin_xyz_wt = \
                grab_xyz_weights(self.model.module.nerf_skin, clone=True)
            self.model.module.feat_xyz_wt = \
                grab_xyz_weights(self.model.module.nerf_feat, clone=True)

        #TODO reset beta
        if opts.reset_beta:
            # 将 nerf_coarse 模型中的 beta 参数的所有元素都设置为了 0.1。
            self.model.module.nerf_coarse.beta.data[:] = 0.1
            # 一个神经辐射场（NeRF）模型的粗糙（coarse）版本

        # start training
        for epoch in range(0, self.num_epochs):
            # 在每个epoch的开始，设置self.model.epoch为当前的epoch，用于跟踪训练进度。
            self.model.epoch = epoch

            # evaluation
            # 清空CUDA缓存以释放未使用的GPU内存。
            torch.cuda.empty_cache()
            # 设置模型的图像尺寸为评估（渲染）大小，opts.render_size。
            self.model.module.img_size = opts.render_size
            # 调用self.eval()函数进行模型评估，可能是在验证集上，得到渲染的序列rendered_seq和辅助序列aux_seq。
            rendered_seq, aux_seq = self.eval()    
            # 将模型的图像尺寸重置为训练尺寸，opts.img_size            
            self.model.module.img_size = opts.img_size
            # 如果是第一个epoch，保存一些相机参数，这通常用于初始化或者确保模型有一个稳定的开始。
            if epoch==0: self.save_network('0') # to save some cameras
            # 如果当前进程是主进程（opts.local_rank==0），将渲染的图像序列添加到TensorBoard日志。
            if opts.local_rank==0: self.add_image_grid(rendered_seq, log, epoch)

            # 调用self.reset_hparams(epoch)函数重置或更新某些超参数，可能基于当前的epoch。
            self.reset_hparams(epoch)

            # 再次清空CUDA缓存。
            torch.cuda.empty_cache()
            
            ## TODO harded coded
            #if opts.freeze_proj:
            #    if self.model.module.progress<0.8:
            #        #opts.nsample=64
            #        opts.ndepth=2
            #    else:
            #        #opts.nsample = nsample
            #        opts.ndepth = self.model.module.ndepth_bk

            # 进行实际的训练，调用self.train_one_epoch(epoch, log)函数执行一个epoch的训练。
            self.train_one_epoch(epoch, log)
    
            # 打印保存模型的信息
            print('saving the model at the end of epoch {:d}, iters {:d}'.\
                              format(epoch, self.model.module.total_steps))
            # 保存一个最新的（'latest'）模型状态
            self.save_network('latest')
            # 保存一个以epoch编号命名的模型状态
            self.save_network(str(epoch+1))

            '''
            按照 self.num_epochs 指定的次数执行多个 epoch。self.num_epochs 是在代码的其他部分设置的，
            这个值定义了总共需要进行多少个训练周期。这个循环确保模型会经过多次迭代的训练，
            每次迭代都可能更新模型的权重，以改进其性能。
            
            '''

            '''
            此代码块的目的是进行模型的训练，同时在每个epoch结束后进行评估，保存模型状态，并在TensorBoard上记录训练过程。
            这是一个典型的训练循环，确保模型在每个epoch之后得到保存和评估，以便监控训练进度和性能。
            '''

    @staticmethod
    def save_cams(opts,aux_seq, save_prefix, latest_vars,datasets, evalsets, obj_scale,
            trainloader=None, unc_filter=True):
        """
        save cameras to dir and modify dataset 
        """
        mkdir_p(save_prefix)
        #训练集和评估集
        dataset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in datasets}
        evalset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in evalsets}
        #这段代码创建了一个名为 line_dict 的字典，
        #它将训练数据集中每个数据集的名称（通常是目录名）映射到相应的数据集对象。
        if trainloader is not None:
            line_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in trainloader}

        #获取辅助序列中图像路径的数量
        length = len(aux_seq['impath'])
        #获取一个布尔数组，表示每个图像是否有效。
        valid_ids = aux_seq['is_valid']
        idx_combine = 0
        for i in range(length):
            #获取图像路径和对应的序列名称。
            impath = aux_seq['impath'][i]
            #获取图像对应的序列名称。
            seqname = impath.split('/')[-2]
            #获取相机的姿态矩阵 rtk。
            rtk = aux_seq['rtk'][i]

            #如果 `unc_filter` 设置为 `True`，这意味着在处理相机参数（特别是相机的旋转部分）时，代码会实施一个过滤机制，
            #以确保使用的相机参数是有效的。在某些情况下，相机追踪或估计算法可能会失败，导致某些帧的相机参数不准确或无效。
            #这通常发生在视觉不清晰或运动模糊强烈的帧。
            #为了解决这个问题，代码会在同一序列中寻找最近的有效帧，并将当前帧的旋转矩阵替换为该有效帧的旋转矩阵。
            #这种替换假设在两帧之间，相机的旋转不会有大的变化，所以一个有效帧的旋转矩阵可以作为无效帧的合理近似。
            #这个步骤的目的是提高相机参数的整体质量，尤其是在自动化处理或长序列处理中，
            #这样可以减少错误的相机参数对后续处理步骤（如三维重建、动画或其他计算视觉任务）的影响。       
            if unc_filter:
                # in the same sequance find the closest valid frame and replace it
                seq_idx = np.asarray([seqname == i.split('/')[-2] \
                        for i in aux_seq['impath']])
                valid_ids_seq = np.where(valid_ids * seq_idx)[0]
                if opts.local_rank==0 and i==0: 
                    print('%s: %d frames are valid'%(seqname, len(valid_ids_seq)))
                if len(valid_ids_seq)>0 and not aux_seq['is_valid'][i]:
                    closest_valid_idx = valid_ids_seq[np.abs(i-valid_ids_seq).argmin()]
                    rtk[:3,:3] = aux_seq['rtk'][closest_valid_idx][:3,:3]

            # 这段代码的目的是根据输入的近-远平面（near-far plane）对相机的平移参数进行缩放，
            # 并且将更新后的相机参数保存到文件中。

            # rescale translation according to input near-far plane
            # 缩放平移向量
            rtk[:3,3] = rtk[:3,3]*obj_scale
            rtklist = dataset_dict[seqname].rtklist
            idx = int(impath.split('/')[-1].split('.')[-2])
            save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx)
            # 保存相机参数文件 
            np.savetxt(save_path, rtk)
            # 更新相机参数列表 
            rtklist[idx] = save_path
            # 更新数据集字典
            evalset_dict[seqname].rtklist[idx] = save_path
            # 条件性更新训练加载器
            if trainloader is not None:
                line_dict[seqname].rtklist[idx] = save_path
            
            #save to rtraw 
            #将相机的旋转和平移矩阵
            latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
            #将相机的旋转部分单独保存到数组
            #这里的 idx_combine 是一个索引，用于跟踪 latest_vars 字典中的位置，
            #这个字典可能用于存储整个数据集的相机参数。
            latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]

            if idx==len(rtklist)-2:
                # to cover the last
                # 生成并保存最后一帧的相机参数文件路径 save_path。
                save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx+1)
                # 打印出正在写入的相机参数文件的路径（如果当前的进程 local_rank 是 0，通常表示主进程或首个进程）
                if opts.local_rank==0: print('writing cam %s'%save_path)
                # 使用 np.savetxt 将相机参数 rtk 保存到文本文件中。
                np.savetxt(save_path, rtk)
                # 更新 rtklist、evalset_dict[seqname].rtklist 和 line_dict[seqname].rtklist（如果 trainloader 不为空）数组，
                # 使其包含最后一帧的相机参数文件路径。
                rtklist[idx+1] = save_path
                evalset_dict[seqname].rtklist[idx+1] = save_path
                if trainloader is not None:
                    line_dict[seqname].rtklist[idx+1] = save_path
                # 将最后一帧的相机参数保存到 latest_vars['rt_raw'] 和 latest_vars['rtk'] 字典中，确保 idx_combine 递增以继续跟踪。
                idx_combine += 1
                latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
                latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]
            idx_combine += 1
    # 这样做可以确保数据集的每一帧都有相对应的相机参数文件，并且所有的参数都被记录和保存，
    # 以备后续的处理和训练使用。在计算机视觉和三维重建中，相机参数对于正确地将二维图像映射到三维空间非常关键。
        
   # 这个函数 extract_cams 是为了提取和保存相机参数。     
    def extract_cams(self, full_loader):
        # store cameras
        # 设定 opts 变量，这通常包含了用于指定训练选项和配置的参数.
        opts = self.opts
        # 确定要评估的帧的索引范围，这里是 evalloader 的长度，即评估数据加载器中的所有帧。
        idx_render = range(len(self.evalloader))
        chunk = 50
        # 设置处理相机参数的批次大小（chunk），这里是 50，表示一次处理 50 帧的相机参数
        aux_seq = []
        # 循环遍历所有帧，每次处理一个批次的帧，并将结果存储在 aux_seq 列表中。
        for i in range(0, len(idx_render), chunk):
            aux_seq.append(self.eval_cam(idx_render=idx_render[i:i+chunk]))
        # 使用 merge_dict 函数将多个批次的结果合并到一个字典中。
        aux_seq = merge_dict(aux_seq)
        # 将字典中的某些键对应的列表转换为 NumPy 数组，包括 rtk（相机参数），
        # kaug（可能是相机内参的调整），masks（遮罩），is_valid（有效帧的标记），err_valid（有效帧的误差）。
        aux_seq['rtk'] = np.asarray(aux_seq['rtk'])
        aux_seq['kaug'] = np.asarray(aux_seq['kaug'])
        aux_seq['masks'] = np.asarray(aux_seq['masks'])
        aux_seq['is_valid'] = np.asarray(aux_seq['is_valid'])
        aux_seq['err_valid'] = np.asarray(aux_seq['err_valid'])

        # 设置保存相机参数的路径前缀 save_prefix。
        save_prefix = '%s/init-cam'%(self.save_dir)
        trainloader=self.trainloader.dataset.datasets
        # 调用 save_cams 函数，将提取的相机参数保存到指定的路径，并更新数据加载器中的相应信息。
        self.save_cams(opts,aux_seq, save_prefix,
                    self.model.module.latest_vars,
                    full_loader.dataset.datasets,
                self.evalloader.dataset.datasets,
                self.model.obj_scale, trainloader=trainloader,
                unc_filter=opts.unc_filter)
        # 使用 dist.barrier() 确保在所有进程中此步骤都已完成，这是并行或分布式训练时同步各个进程的常见做法。
        dist.barrier() # wait untail all have finished

        # 如果当前进程的 local_rank 是 0（通常是主进程），它将调用 render_root_txt 函数为每个数据集绘制相机轨迹。
        if opts.local_rank==0:
            # draw camera trajectory
            for dataset in full_loader.dataset.datasets:
                seqname = dataset.imglist[0].split('/')[-2]
                render_root_txt('%s/%s-'%(save_prefix,seqname), 0)
    # 这个函数的目的是在训练之前提取和初始化相机参数，这对于后续的三维重建和渲染是非常关键的。
    # 通过保存相机参数，模型能够利用这些参数将二维图像正确映射到三维空间，这对于产生准确的三维重建非常重要。

    # 这个 reset_nf 函数的目的是设置或重置模型的近平面和远平面（near-far plane），
    # 这是三维渲染中用于确定视锥（view frustum）的参数。
    def reset_nf(self):
        opts = self.opts
        # save near-far plane
        # 计算形状顶点的界限（shape_verts），这些顶点在单位范围内，
        # 通过乘以近远平面的平均值并扩大 20% 来确定对象的大致大小。
        shape_verts = self.model.dp_verts_unit / 3 * self.model.near_far.mean()
        shape_verts = shape_verts * 1.2
        # save object bound if first stage
        # 如果是训练的第一阶段（即没有预训练模型），
        # 并且 bound_factor 大于 0，那么会按照 bound_factor 进一步扩展形状顶点的界限。
        if opts.model_path=='' and opts.bound_factor>0:
            shape_verts = shape_verts*opts.bound_factor
            # 将计算出的对象界限保存到模型的最新变量中（latest_vars['obj_bound']）
            self.model.module.latest_vars['obj_bound'] = \
            shape_verts.abs().max(0)[0].detach().cpu().numpy()
        # 如果当前的近远平面没有有效值（由 self.model.near_far[:,0].sum()==0 检查），
        # 则调用 get_near_far 函数来计算新的近远平面值。
        if self.model.near_far[:,0].sum()==0: # if no valid nf plane loaded
            self.model.near_far.data = get_near_far(self.model.near_far.data,
                                                self.model.latest_vars,
                                         pts=shape_verts.detach().cpu().numpy())
        # 将计算出的近远平面值保存到指定的路径（save_path）。
        save_path = '%s/init-nf.txt'%(self.save_dir)
        # 将近远平面值与对象的缩放比例（self.model.obj_scale）相乘，然后使用 np.savetxt 保存到文本文件中。
        save_nf = self.model.near_far.data.cpu().numpy() * self.model.obj_scale
        np.savetxt(save_path, save_nf)
    # 这个函数确保了在模型的训练或使用过程中，近远平面的设置是根据模型当前的状态和数据来确定的，
    # 从而保持了渲染过程的一致性和准确性。

    # 这个 warmup_shape 函数是在神经网络训练流程中对模型的形状进行预热的步骤。
    # 在训练深度学习模型时，预热（warmup）通常指的是在开始进行正式的训练前先进行一段时间的训练，以此来初始化模型的权重，
    # 使其达到一个较好的起始状态，从而有助于模型的收敛和泛化性能。
    # 这里的“形状预热”指的可能是模型在训练前先对输入数据的形状或结构进行学习，以适应数据的分布。
    def warmup_shape(self, log):
        opts = self.opts

        # force using warmup forward, dataloader, cnn root
        # 强制模型使用预热的前向传播函数 forward_warmup_shape。
        # 这可能是一个专为预热设计的前向传播函数，与默认的前向传播函数 forward_default 有所不同。
        self.model.module.forward = self.model.module.forward_warmup_shape
        full_loader = self.trainloader  # store original loader
        # 临时修改数据加载器 trainloader，使其仅包含200个数据点，这可能是为了加快预热阶段的训练速度。
        self.trainloader = range(200)
        # 设置预热阶段的训练周期数 self.num_epochs 为 opts.warmup_shape_ep。
        self.num_epochs = opts.warmup_shape_ep

        # training
        # 调用 init_training 函数来初始化训练设置。
        self.init_training()
        for epoch in range(0, opts.warmup_shape_ep):
            self.model.epoch = epoch
            # 进行预热阶段的训练，对每个epoch调用 train_one_epoch 函数，
            # 并将 warmup 参数设置为 True，可能意味着在这一阶段会应用不同的训练策略或参数。
            self.train_one_epoch(epoch, log, warmup=True)
            # 在每个epoch结束后，调用 save_network 函数来保存当前模型的状态，文件名前缀为 'mlp-'。
            self.save_network(str(epoch+1), 'mlp-') 

        # restore dataloader, rts, forward function
        # 预热结束后，将前向传播函数、数据加载器和训练周期数重置为默认值。
        self.model.module.forward = self.model.module.forward_default
        self.trainloader = full_loader
        self.num_epochs = opts.num_epochs

        # start from low learning rate again
        # 再次调用 init_training 函数来重置训练设置，
        # 并将模型的总步数 total_steps 和进度 progress 重置为0，以便开始正式的训练。
        self.init_training()
        self.model.module.total_steps = 0
        self.model.module.progress = 0.
    # 整个预热过程是为了让模型在进入更复杂的训练阶段之前，对数据的基本形状有一个较好的适应性和理解。

    # 这个 warmup_pose 函数是在神经网络训练流程中对模型的姿态（pose）进行预热的步骤。在训练深度学习模型时，
    # 特别是在处理图像或视频数据时，对姿态的预热是为了让模型学会如何从数据中提取和理解姿态信息。
    def warmup_pose(self, log, pose_cnn_path):
        opts = self.opts

        # force using warmup forward, dataloader, cnn 
        # 设置模型使用基于CNN的根部姿态表示 root_basis，
        self.model.module.root_basis = 'cnn'
        # 并且暂时不使用相机参数 use_cam。
        self.model.module.use_cam = False
        # 置模型使用专门为预热设计的前向传播函数 forward_warmup。
        self.model.module.forward = self.model.module.forward_warmup
        full_loader = self.dataloader  # store original loader
        # 临时修改数据加载器 dataloader，使其仅包含200个数据点。
        self.dataloader = range(200)
        # 保存原始的根部姿态表示 nerf_root_rts 并使用一个新的表示 dp_root_rts 进行预热训练。
        original_rp = self.model.module.nerf_root_rts
        self.model.module.nerf_root_rts = self.model.module.dp_root_rts
        # 删除 dp_root_rts 以节省内存。
        del self.model.module.dp_root_rts
        # 设置预热阶段的训练周期数为 opts.warmup_pose_ep，并标记为姿态预热阶段。
        self.num_epochs = opts.warmup_pose_ep
        self.model.module.is_warmup_pose=True

        # 如果没有提供预训练的CNN模型路径，则进行姿态预热的训练。
        if pose_cnn_path=='':
            # training
            # 初始化训练设置。
            self.init_training()
            for epoch in range(0, opts.warmup_pose_ep):
                self.model.epoch = epoch
                # 对每个epoch调用 train_one_epoch 函数，并将 warmup 参数设置为 True。
                self.train_one_epoch(epoch, log, warmup=True)
                # 在每个epoch结束后，保存当前模型状态，文件名前缀为 'cnn-'。
                self.save_network(str(epoch+1), 'cnn-') 

                # eval
                #_,_ = self.model.forward_warmup(None)
                # rendered_seq = self.model.warmup_rendered 
                # if opts.local_rank==0: self.add_image_grid(rendered_seq, log, epoch)

        # 如果提供了预训练的CNN模型路径，则直接加载这些状态。
        else: 
            pose_states = torch.load(opts.pose_cnn_path, map_location='cpu')
            pose_states = self.rm_module_prefix(pose_states, 
                    prefix='module.nerf_root_rts')
            self.model.module.nerf_root_rts.load_state_dict(pose_states, 
                                                        strict=False)

        # extract camera and near far planes
        self.extract_cams(full_loader)

        # restore dataloader, rts, forward function
        # 恢复原始的数据加载器、根部姿态表示和前向传播函数。
        self.model.module.root_basis=opts.root_basis
        self.model.module.use_cam = opts.use_cam
        self.model.module.forward = self.model.module.forward_default
        self.dataloader = full_loader
        # 删除临时使用的根部姿态表示 nerf_root_rts，恢复原始的 nerf_root_rts。
        del self.model.module.nerf_root_rts
        self.model.module.nerf_root_rts = original_rp
        # 重置训练周期数和模型的其他训练设置。
        self.num_epochs = opts.num_epochs
        self.model.module.is_warmup_pose=False

        # start from low learning rate again
        # 再次调用 init_training 函数来重置训练设置，
        # 并将模型的总步数 total_steps 和进度 progress 重置为0，以便开始正式的训练。
        self.init_training()
        self.model.module.total_steps = 0
        self.model.module.progress = 0.

    # 这个函数 train_one_epoch 是定义在训练类中的一个方法，负责执行模型训练过程中的一个完整的 epoch。
    # 函数接收 epoch（当前训练的轮数），log（用于记录训练日志的对象），以及一个标记 warmup（表示是否处于预热阶段）。

    def train_one_epoch(self, epoch, log, warmup=False):
        """
        training loop in a epoch
        """
        opts = self.opts
        # 设置模型为训练模式。
        self.model.train()
        dataloader = self.trainloader

        # 如果不是预热阶段，会设置数据加载器的采样器，以便打乱数据。
        if not warmup: dataloader.sampler.set_epoch(epoch) # necessary for shuffling

        # 遍历数据加载器中的批次数据，如果达到了200次批处理乘以累积步骤数 opts.accu_steps，则中断循环。
        for i, batch in enumerate(dataloader):
            if i==200*opts.accu_steps:
                break
            # 如果开启了调试模式，会记录和输出数据加载时间。
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('load time:%.2f'%(time.time()-start_time))
            # 如果不是预热阶段，会更新模型的进度指示器，选择损失指示器，并更新不同部分（如根、身体、形状等）的训练指示器。
            if not warmup:
                self.model.module.progress = float(self.model.total_steps) /\
                                               self.model.final_steps
                self.select_loss_indicator(i)
                self.update_root_indicator(i)
                self.update_body_indicator(i)
                self.update_shape_indicator(i)
                self.update_cvf_indicator(i)

#                rtk_all = self.model.module.compute_rts()
#                self.model.module.rtk_all = rtk_all.clone()
#
#            # change near-far plane for all views
#            if self.model.module.progress>=opts.nf_reset:
#                rtk_all = rtk_all.detach().cpu().numpy()
#                valid_rts = self.model.module.latest_vars['idk'].astype(bool)
#                self.model.module.latest_vars['rtk'][valid_rts,:3] = rtk_all[valid_rts]
#                self.model.module.near_far.data = get_near_far(
#                                              self.model.module.near_far.data,
#                                              self.model.module.latest_vars)
#
#            self.optimizer.zero_grad()
            total_loss,aux_out = self.model(batch)
            # 计算损失（total_loss）并除以累积步骤数（accu_steps）进行归一化。
            total_loss = total_loss/self.accu_steps

            # 如果是调试模式，记录前向传播时间。
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('forward time:%.2f'%(time.time()-start_time))
            # 执行反向传播（.backward()），进行梯度计算。
            total_loss.mean().backward()
            # 如果是调试模式，记录反向传播的时间。
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('forward back time:%.2f'%(time.time()-start_time))

            # 每经过 accu_steps 次迭代，进行梯度裁剪和优化器步骤，并更新学习率调度器。
            if (i+1)%self.accu_steps == 0:
                self.clip_grad(aux_out)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                # 如果发现根部姿态表示的梯度大于某个阈值，并且已经执行了足够的总步数，
                # 则会从保存的最新模型中重新加载参数，这是为了防止梯度爆炸或者不稳定的训练情况。
                if aux_out['nerf_root_rts_g']>1*opts.clip_scale and \
                                self.model.total_steps>200*self.accu_steps:
                    latest_path = '%s/params_latest.pth'%(self.save_dir)
                    self.load_network(latest_path, is_eval=False, rm_prefix=False)
                
            for i,param_group in enumerate(self.optimizer.param_groups):
                aux_out['lr_%02d'%i] = param_group['lr']

            self.model.module.total_steps += 1
            # 减少冻结和重新骨化的计数器（counter_frz_rebone）。
            self.model.module.counter_frz_rebone -= 1./self.model.final_steps
            aux_out['counter_frz_rebone'] = self.model.module.counter_frz_rebone

            # 如果是主进程（opts.local_rank==0），则保存日志。
            if opts.local_rank==0: 
                self.save_logs(log, aux_out, self.model.module.total_steps, 
                        epoch)
            # 如果是调试模式，记录每个总步骤的时间，并同步 CUDA 设备。
            if opts.debug:
                if 'start_time' in locals().keys():
                    torch.cuda.synchronize()
                    print('total step time:%.2f'%(time.time()-start_time))
                torch.cuda.synchronize()
                start_time = time.time()

    # 整个函数的目的是在训练过程中按批次迭代数据，更新模型参数，并记录训练进度。通过传入的 warmup 参数，
    # 它可以区分是否处于预热阶段，从而调整训练行为。例如，在预热阶段可能不会更新学习率或者不会执行某些特定的更新策略。


    # 这个函数 update_cvf_indicator 是一个训练类中的一部分，
    # 用于控制训练过程中是否更新规范体积特征（Canonical Volume Features，简称 CVF）。
    # CVF 在神经渲染或 3D 重建框架中可能代表场景体积的某种持久和规范的特征表示。

    def update_cvf_indicator(self, i):
        """
        whether to update canoical volume features
        0: update all
        1: freeze 
        """
        opts = self.opts

        # during kp reprojection optimization
        # 关键点重投影优化期间：
        # 如果选项 opts 中的 freeze_proj 设置为 True，并且模型的进度在 proj_start 和 proj_end 之间，
        # 此时应该冻结 CVF，不进行更新（将 cvf_update 设置为 1）。
        # 这段时间可能是专门用来优化关键点重投影的，更新 CVF 可能会对此过程产生负面影响。
        if (opts.freeze_proj and self.model.module.progress >= opts.proj_start and \
               self.model.module.progress < (opts.proj_start+opts.proj_end)):
            self.model.module.cvf_update = 1
        else:
            self.model.module.cvf_update = 0
        
        # freeze shape after rebone      
        # 如果 counter_frz_rebone 大于 0，表明模型处于重新定义骨架之后的一个时期，
        # 在这个时期内应该冻结形状，因此也不更新 CVF（将 cvf_update 设置为 1）。  
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.cvf_update = 1
        # 冻结 CVF：如果选项 opts 中的 freeze_cvf 设置为 True，
        # 则在训练期间应该冻结 CVF，不进行更新（将 cvf_update 设置为 1）。
        if opts.freeze_cvf:
            self.model.module.cvf_update = 1
    
    # 当 cvf_update 设置为 0 时，表示 CVF 可以在训练过程中更新。这个标志用于根据训练的当前阶段或正在执行的特定操作，
    # 有选择地冻结或更新 CVF，从而允许对训练过程进行更多的控制，并可能获得更好的训练结果。

    # 这个函数 update_shape_indicator 是神经网络训练过程中的一部分，用于控制模型形状参数是否应该被更新。
    # 在训练深度学习模型时，有时需要冻结网络的某些部分以防止它们在特定阶段被更新，这就是所谓的冻结（freeze）。
    # 这个函数根据训练的不同阶段和设定的条件，动态决定是否更新（或冻结）模型的形状参数。

    def update_shape_indicator(self, i):
        """
        whether to update shape
        0: update all
        1: freeze shape
        """
        opts = self.opts
        # incremental optimization
        # or during kp reprojection optimization
        # 如果模型已经预加载了权重（opts.model_path != ''），
        # 且训练进度（self.model.module.progress）小于预热步数（opts.warmup_steps），
        # 或者如果设置了冻结投影参数（opts.freeze_proj）并且训练进度在投影的起始和结束阶段（opts.proj_start 
        # 到 opts.proj_end）之间，
        # 这时应该冻结形状更新（self.model.module.shape_update = 1）。
        if (opts.model_path!='' and \
        self.model.module.progress < opts.warmup_steps)\
         or (opts.freeze_proj and self.model.module.progress >= opts.proj_start and \
               self.model.module.progress <(opts.proj_start + opts.proj_end)):
            self.model.module.shape_update = 1
        else:
            self.model.module.shape_update = 0

        # freeze shape after rebone
        # 如果 counter_frz_rebone 大于 0，表明模型处于重新定义骨架之后的一个时期，
        # 在这个时期内应该冻结形状更新（self.model.module.shape_update = 1）。   
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.shape_update = 1
        # 如果设置了 opts.freeze_shape，表示在整个训练过程中都要冻结形状参数（self.model.module.shape_update = 1）。
        if opts.freeze_shape:
            self.model.module.shape_update = 1
    # 当 self.model.module.shape_update 设置为 0 时，表示形状参数可以在训练过程中更新。
    # 这个标志用于根据训练的当前阶段或正在执行的特定操作，
    # 有选择性地冻结或更新形状参数，从而可以更精细地控制训练过程。

    # 函数 update_root_indicator 的作用是控制模型根姿态（即整体的位置和旋转，通常是指模型的全局变换）
    # 是否应该在训练过程中更新。
    # 这种控制对于训练稳定性或确保训练的特定阶段专注于特定的参数非常重要。

    def update_root_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        # 如果设置了投影参数的冻结（opts.freeze_proj）并且启用了根姿态稳定（opts.root_stab）：
        # 当模型的训练进度（self.model.module.progress）在冻结根姿态开始（opts.frzroot_start）
        # 和投影结束后的短暂阶段（opts.proj_start + opts.proj_end + 0.01）之间时，
        # 根姿态更新被冻结（self.model.module.root_update = 0）。这通常是为了在进行关键点重投影优化时保持根姿态的稳定。
        if (opts.freeze_proj and \
            opts.root_stab and \
           self.model.module.progress >=(opts.frzroot_start) and \
           self.model.module.progress <=(opts.proj_start + opts.proj_end+0.01))\
           : # to stablize
            self.model.module.root_update = 0
        else:
            self.model.module.root_update = 1
        
        # freeze shape after rebone
        # 如果 counter_frz_rebone 的计数大于 0，这表明模型正处于一个特定的训练阶段，
        # 在这个阶段应该冻结根姿态更新（self.model.module.root_update = 0）。
        if self.model.module.counter_frz_rebone > 0:
            self.model.module.root_update = 0
        
        if opts.freeze_root: # to stablize
            self.model.module.root_update = 0
    # 函数 update_body_indicator 的作用是在训练过程中控制模型的身体部位是否应该更新。
    # 这对于冻结训练的某些部分，以便训练集中于特定的参数或部分，是非常有用的。

    def update_body_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        
        # 如果设置了投影冻结（opts.freeze_proj）：
        # 当模型的训练进度（self.model.module.progress）
        # 小于或等于冻结身体更新的结束点（opts.frzbody_end）时，身体更新将被冻结（self.model.module.body_update = 0）。
        # 这意味着在这个阶段，
        # 模型的身体部分的参数不会更新，这可能是因为开发者希望在训练的早期阶段集中优化其他部分，如根姿态或其他特征。
        if opts.freeze_proj and \
           self.model.module.progress <=opts.frzbody_end: 
            self.model.module.body_update = 0
        
        # 否则：如果不在上述条件中，身体更新标志被设置为 1（self.model.module.body_update = 1），
        # 表示模型的身体部位的参数可以在训练过程中更新。
        else:
            self.model.module.body_update = 1
        # 此函数允许开发者通过控制 body_update 标志来有选择地冻结或更新模型的身体部分。
        # 这样的控制可以帮助在训练的特定阶段内提高稳定性和性能。


    def select_loss_indicator(self, i):
        """
        0: flo
        1: flo/sil/rgb
        """
        opts = self.opts
        if not opts.root_opt or \
            self.model.module.progress > (opts.warmup_steps):
            self.model.module.loss_select = 1
        elif i%2 == 0:
            self.model.module.loss_select = 0
        else:
            self.model.module.loss_select = 1

        #self.model.module.loss_select=1
        

    def reset_hparams(self, epoch):
        """
        reset hyper-parameters based on current geometry / cameras
        """
        opts = self.opts
        mesh_rest = self.model.latest_vars['mesh_rest']

        # reset object bound, for feature matching
        if epoch>int(self.num_epochs*(opts.bound_reset)):
            if mesh_rest.vertices.shape[0]>100:
                self.model.latest_vars['obj_bound'] = 1.2*np.abs(mesh_rest.vertices).max(0)
        
        # reinit bones based on extracted surface
        # only reinit for the initialization phase
        if opts.lbs and opts.model_path=='' and \
                        (epoch==int(self.num_epochs*opts.reinit_bone_steps) or\
                         epoch==0 or\
                         epoch==int(self.num_epochs*opts.warmup_steps)//2):
            reinit_bones(self.model.module, mesh_rest, opts.num_bones)
            self.init_training() # add new params to optimizer
            if epoch>0:
                # freeze weights of root pose in the following 1% iters
                self.model.module.counter_frz_rebone = 0.01
                #reset error stats
                self.model.module.latest_vars['fp_err']      [:]=0
                self.model.module.latest_vars['flo_err']     [:]=0
                self.model.module.latest_vars['sil_err']     [:]=0
                self.model.module.latest_vars['flo_err_hist'][:]=0

        # need to add bones back at 2nd opt
        if opts.model_path!='':
            self.model.module.nerf_models['bones'] = self.model.module.bones

        # add nerf-skin when the shape is good
        if opts.lbs and opts.nerf_skin and \
                epoch==int(self.num_epochs*opts.dskin_steps):
            self.model.module.nerf_models['nerf_skin'] = self.model.module.nerf_skin

        self.broadcast()

    def broadcast(self):
        """
        broadcast variables to other models
        """
        dist.barrier()
        if self.opts.lbs:
            dist.broadcast_object_list(
                    [self.model.module.num_bones, 
                    self.model.module.num_bone_used,],
                    0)
            dist.broadcast(self.model.module.bones,0)
            dist.broadcast(self.model.module.nerf_body_rts[1].rgb[0].weight, 0)
            dist.broadcast(self.model.module.nerf_body_rts[1].rgb[0].bias, 0)

        dist.broadcast(self.model.module.near_far,0)
   
    def clip_grad(self, aux_out):
        """
        gradient clipping
        """
        is_invalid_grad=False
        grad_nerf_coarse=[]
        grad_nerf_beta=[]
        grad_nerf_feat=[]
        grad_nerf_beta_feat=[]
        grad_nerf_fine=[]
        grad_nerf_unc=[]
        grad_nerf_flowbw=[]
        grad_nerf_skin=[]
        grad_nerf_vis=[]
        grad_nerf_root_rts=[]
        grad_nerf_body_rts=[]
        grad_root_code=[]
        grad_pose_code=[]
        grad_env_code=[]
        grad_vid_code=[]
        grad_bones=[]
        grad_skin_aux=[]
        grad_ks=[]
        grad_nerf_dp=[]
        grad_csenet=[]
        for name,p in self.model.named_parameters():
            try: 
                pgrad_nan = p.grad.isnan()
                if pgrad_nan.sum()>0: 
                    print(name)
                    is_invalid_grad=True
            except: pass
            if 'nerf_coarse' in name and 'beta' not in name:
                grad_nerf_coarse.append(p)
            elif 'nerf_coarse' in name and 'beta' in name:
                grad_nerf_beta.append(p)
            elif 'nerf_feat' in name and 'beta' not in name:
                grad_nerf_feat.append(p)
            elif 'nerf_feat' in name and 'beta' in name:
                grad_nerf_beta_feat.append(p)
            elif 'nerf_fine' in name:
                grad_nerf_fine.append(p)
            elif 'nerf_unc' in name:
                grad_nerf_unc.append(p)
            elif 'nerf_flowbw' in name or 'nerf_flowfw' in name:
                grad_nerf_flowbw.append(p)
            elif 'nerf_skin' in name:
                grad_nerf_skin.append(p)
            elif 'nerf_vis' in name:
                grad_nerf_vis.append(p)
            elif 'nerf_root_rts' in name:
                grad_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                grad_nerf_body_rts.append(p)
            elif 'root_code' in name:
                grad_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                grad_pose_code.append(p)
            elif 'env_code' in name:
                grad_env_code.append(p)
            elif 'vid_code' in name:
                grad_vid_code.append(p)
            elif 'module.bones' == name:
                grad_bones.append(p)
            elif 'module.skin_aux' == name:
                grad_skin_aux.append(p)
            elif 'module.ks_param' == name:
                grad_ks.append(p)
            elif 'nerf_dp' in name:
                grad_nerf_dp.append(p)
            elif 'csenet' in name:
                grad_csenet.append(p)
            else: continue
        
        # freeze root pose when using re-projection loss only
        if self.model.module.root_update == 0:
            self.zero_grad_list(grad_root_code)
            self.zero_grad_list(grad_nerf_root_rts)
        if self.model.module.body_update == 0:
            self.zero_grad_list(grad_pose_code)
            self.zero_grad_list(grad_nerf_body_rts)
        if self.opts.freeze_body_mlp:
            self.zero_grad_list(grad_nerf_body_rts)
        if self.model.module.shape_update == 1:
            self.zero_grad_list(grad_nerf_coarse)
            self.zero_grad_list(grad_nerf_beta)
            self.zero_grad_list(grad_nerf_vis)
            #TODO add skinning 
            self.zero_grad_list(grad_bones)
            self.zero_grad_list(grad_nerf_skin)
            self.zero_grad_list(grad_skin_aux)
        if self.model.module.cvf_update == 1:
            self.zero_grad_list(grad_nerf_feat)
            self.zero_grad_list(grad_nerf_beta_feat)
            self.zero_grad_list(grad_csenet)
        if self.opts.freeze_coarse:
            # freeze shape
            # this include nerf_coarse, nerf_skin (optional)
            grad_coarse_mlp = []
            grad_coarse_mlp += self.find_nerf_coarse(\
                                self.model.module.nerf_coarse)
            grad_coarse_mlp += self.find_nerf_coarse(\
                                self.model.module.nerf_skin)
            grad_coarse_mlp += self.find_nerf_coarse(\
                                self.model.module.nerf_feat)
            self.zero_grad_list(grad_coarse_mlp)

            #self.zero_grad_list(grad_nerf_coarse) # freeze shape

            # freeze skinning 
            self.zero_grad_list(grad_bones)
            self.zero_grad_list(grad_skin_aux)
            #self.zero_grad_list(grad_nerf_skin) # freeze fine shape

            ## freeze pose mlp
            #self.zero_grad_list(grad_nerf_body_rts)

            # add vis
            self.zero_grad_list(grad_nerf_vis)
            #print(self.model.module.nerf_coarse.xyz_encoding_1[0].weight[0,:])
           
        clip_scale=self.opts.clip_scale
 
        #TODO don't clip root pose
        aux_out['nerf_coarse_g']   = clip_grad_norm_(grad_nerf_coarse,    1*clip_scale)
        aux_out['nerf_beta_g']     = clip_grad_norm_(grad_nerf_beta,      1*clip_scale)
        aux_out['nerf_feat_g']     = clip_grad_norm_(grad_nerf_feat,     .1*clip_scale)
        aux_out['nerf_beta_feat_g']= clip_grad_norm_(grad_nerf_beta_feat,.1*clip_scale)
        aux_out['nerf_fine_g']     = clip_grad_norm_(grad_nerf_fine,     .1*clip_scale)
        aux_out['nerf_unc_g']     = clip_grad_norm_(grad_nerf_unc,       .1*clip_scale)
        aux_out['nerf_flowbw_g']   = clip_grad_norm_(grad_nerf_flowbw,   .1*clip_scale)
        aux_out['nerf_skin_g']     = clip_grad_norm_(grad_nerf_skin,     .1*clip_scale)
        aux_out['nerf_vis_g']      = clip_grad_norm_(grad_nerf_vis,      .1*clip_scale)
        aux_out['nerf_root_rts_g'] = clip_grad_norm_(grad_nerf_root_rts,100*clip_scale)
        aux_out['nerf_body_rts_g'] = clip_grad_norm_(grad_nerf_body_rts,100*clip_scale)
        aux_out['root_code_g']= clip_grad_norm_(grad_root_code,          .1*clip_scale)
        aux_out['pose_code_g']= clip_grad_norm_(grad_pose_code,         100*clip_scale)
        aux_out['env_code_g']      = clip_grad_norm_(grad_env_code,      .1*clip_scale)
        aux_out['vid_code_g']      = clip_grad_norm_(grad_vid_code,      .1*clip_scale)
        aux_out['bones_g']         = clip_grad_norm_(grad_bones,          1*clip_scale)
        aux_out['skin_aux_g']   = clip_grad_norm_(grad_skin_aux,         .1*clip_scale)
        aux_out['ks_g']            = clip_grad_norm_(grad_ks,            .1*clip_scale)
        aux_out['nerf_dp_g']       = clip_grad_norm_(grad_nerf_dp,       .1*clip_scale)
        aux_out['csenet_g']        = clip_grad_norm_(grad_csenet,        .1*clip_scale)

        #if aux_out['nerf_root_rts_g']>10:
        #    is_invalid_grad = True
        if is_invalid_grad:
            self.zero_grad_list(self.model.parameters())
            
    @staticmethod
    def find_nerf_coarse(nerf_model):
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
                # get the weights according to coarse posec
                # 63 = 3 + 60
                # 60 = (num_freqs, 2, 3)
                out_dim = p.shape[0]
                pos_dim = nerf_model.in_channels_xyz-nerf_model.in_channels_code
                # TODO
                num_coarse = 8 # out of 10
                #num_coarse = 10 # out of 10
                #num_coarse = 1 # out of 10
           #     p.grad[:,:3] = 0 # xyz
           #     p.grad[:,3:pos_dim].view(out_dim,-1,6)[:,:num_coarse] = 0 # xyz-coarse
                p.grad[:,pos_dim:] = 0 # others
            else:
                param_list.append(p)
        return param_list

    @staticmethod 
    def render_vid(model, batch):
        # 获取模型的配置选项。
        opts=model.opts
        # 模型设置输入数据批次。
        model.set_input(batch)
        # 获取模型的相机姿态参数（通常是旋转矩阵和平移向量）。
        rtk = model.rtk
        # 克隆相机内参矩阵的增强版本，这些内参矩阵可能已经根据某些条件进行了调整或优化。
        kaug=model.kaug.clone()
        # 获取嵌入ID，这可能是用于查找嵌入式特征的索引或键。
        embedid=model.embedid

        # 调用 nerf_render 函数渲染图像，使用提供的相机姿态参数、相机内参和嵌入ID，ndepth 参数指定渲染时使用的深度层数。
        rendered, _ = model.nerf_render(rtk, kaug, embedid, ndepth=opts.ndepth)
        # 从 rendered 字典中删除不需要的可视化关键字
        if 'xyz_camera_vis' in rendered.keys():    del rendered['xyz_camera_vis']   
        if 'xyz_canonical_vis' in rendered.keys(): del rendered['xyz_canonical_vis']
        if 'pts_exp_vis' in rendered.keys():       del rendered['pts_exp_vis']      
        if 'pts_pred_vis' in rendered.keys():      del rendered['pts_pred_vis'] 
        # 初始化一个空字典来存储最终的渲染结果。    
        rendered_first = {}
        # 迭代 rendered 字典的条目：对于字典中的每个条目，如果它是一个多维张量，只取批次大小的一半。
        # 这可能是为了去除用于损失计算的术语，或者只是为了从可能是成对数据的集合中获取第一组数据。
        for k,v in rendered.items():
            if v.dim()>0: 
                bs=v.shape[0]
                rendered_first[k] = v[:bs//2] # remove loss term
        return rendered_first 
        # 函数返回的 rendered_first 字典包含每个键的前半部分数据，这可能是用于视频渲染中的实际图像帧。

    @staticmethod
    def extract_mesh(model,chunk,grid_size,
                      #threshold = -0.005,
                      threshold = -0.002,
                      #threshold = 0.,
                      embedid=None,
                      mesh_dict_in=None):
        opts = model.opts
        mesh_dict = {}
        if model.near_far is not None: 
            bound = model.latest_vars['obj_bound']
        else: bound=1.5*np.asarray([1,1,1])

        if mesh_dict_in is None:
            ptx = np.linspace(-bound[0], bound[0], grid_size).astype(np.float32)
            pty = np.linspace(-bound[1], bound[1], grid_size).astype(np.float32)
            ptz = np.linspace(-bound[2], bound[2], grid_size).astype(np.float32)
            query_yxz = np.stack(np.meshgrid(pty, ptx, ptz), -1)  # (y,x,z)
            #pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
            #query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
            query_yxz = torch.Tensor(query_yxz).to(model.device).view(-1, 3)
            query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)
            query_dir = torch.zeros_like(query_xyz)

            bs_pts = query_xyz.shape[0]
            out_chunks = []
            for i in range(0, bs_pts, chunk):
                query_xyz_chunk = query_xyz[i:i+chunk]
                query_dir_chunk = query_dir[i:i+chunk]

                # backward warping 
                if embedid is not None and not opts.queryfw:
                    query_xyz_chunk, mesh_dict = warp_bw(opts, model, mesh_dict, 
                                                   query_xyz_chunk, embedid)
                if opts.symm_shape: 
                    #TODO set to x-symmetric
                    query_xyz_chunk[...,0] = query_xyz_chunk[...,0].abs()
                xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                out_chunks += [model.nerf_coarse(xyz_embedded, sigma_only=True)]
            vol_o = torch.cat(out_chunks, 0)
            vol_o = vol_o.view(grid_size, grid_size, grid_size)
            #vol_o = F.softplus(vol_o)

            if not opts.full_mesh:
                #TODO set density of non-observable points to small value
                if model.latest_vars['idk'].sum()>0:
                    vis_chunks = []
                    for i in range(0, bs_pts, chunk):
                        query_xyz_chunk = query_xyz[i:i+chunk]
                        if opts.nerf_vis:
                            # this leave no room for halucination and is not what we want
                            xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                            vis_chunk_nerf = model.nerf_vis(xyz_embedded)
                            vis_chunk = vis_chunk_nerf[...,0].sigmoid()
                        else:
                            #TODO deprecated!
                            vis_chunk = compute_point_visibility(query_xyz_chunk.cpu(),
                                             model.latest_vars, model.device)[None]
                        vis_chunks += [vis_chunk]
                    vol_visi = torch.cat(vis_chunks, 0)
                    vol_visi = vol_visi.view(grid_size, grid_size, grid_size)
                    vol_o[vol_visi<0.5] = -1

            ## save color of sampled points 
            #cmap = cm.get_cmap('cool')
            ##pts_col = cmap(vol_visi.float().view(-1).cpu())
            #pts_col = cmap(vol_o.sigmoid().view(-1).cpu())
            #mesh = trimesh.Trimesh(query_xyz.view(-1,3).cpu(), vertex_colors=pts_col)
            #mesh.export('0.obj')
            #pdb.set_trace()

            print('fraction occupied:', (vol_o > threshold).float().mean())
            vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
            vertices = (vertices - grid_size/2)/grid_size*2*bound[None, :]
            mesh = trimesh.Trimesh(vertices, triangles)

            # mesh post-processing 
            if len(mesh.vertices)>0:
                if opts.use_cc:
                    # keep the largest mesh
                    mesh = [i for i in mesh.split(only_watertight=False)]
                    mesh = sorted(mesh, key=lambda x:x.vertices.shape[0])
                    mesh = mesh[-1]

                # assign color based on canonical location
                vis = mesh.vertices
                try:
                    model.module.vis_min = vis.min(0)[None]
                    model.module.vis_len = vis.max(0)[None] - vis.min(0)[None]
                except: # test time
                    model.vis_min = vis.min(0)[None]
                    model.vis_len = vis.max(0)[None] - vis.min(0)[None]
                vis = vis - model.vis_min
                vis = vis / model.vis_len
                if not opts.ce_color:
                    vis = get_vertex_colors(model, mesh, frame_idx=0)
                mesh.visual.vertex_colors[:,:3] = vis*255

        # forward warping
        if embedid is not None and opts.queryfw:
            mesh = mesh_dict_in['mesh'].copy()
            vertices = mesh.vertices
            vertices, mesh_dict = warp_fw(opts, model, mesh_dict, 
                                           vertices, embedid)
            mesh.vertices = vertices
               
        mesh_dict['mesh'] = mesh
        return mesh_dict

    def save_logs(self, log, aux_output, total_steps, epoch):
        for k,v in aux_output.items():
            self.add_scalar(log, k, aux_output,total_steps)
        
    def add_image_grid(self, rendered_seq, log, epoch):
        for k,v in rendered_seq.items():
            grid_img = image_grid(rendered_seq[k],3,3)
            if k=='depth_rnd':scale=True
            elif k=='occ':scale=True
            elif k=='unc_pred':scale=True
            elif k=='proj_err':scale=True
            elif k=='feat_err':scale=True
            else: scale=False
            self.add_image(log, k, grid_img, epoch, scale=scale)

    def add_image(self, log,tag,timg,step,scale=True):
        """
        timg, h,w,x
        """

        if self.isflow(tag):
            timg = timg.detach().cpu().numpy()
            timg = flow_to_image(timg)
        elif scale:
            timg = (timg-timg.min())/(timg.max()-timg.min())
        else:
            timg = torch.clamp(timg, 0,1)
    
        if len(timg.shape)==2:
            formats='HW'
        elif timg.shape[0]==3:
            formats='CHW'
            print('error'); pdb.set_trace()
        else:
            formats='HWC'

        log.add_image(tag,timg,step,dataformats=formats)

    @staticmethod
    def add_scalar(log,tag,data,step):
        if tag in data.keys():
            log.add_scalar(tag,  data[tag], step)

    @staticmethod
    def del_key(states, key):
        if key in states.keys():
            del states[key]
    
    @staticmethod
    def isflow(tag):
        flolist = ['flo_coarse', 'fdp_coarse', 'flo', 'fdp', 'flo_at_samp']
        if tag in flolist:
           return True
        else:
            return False

    @staticmethod
    def zero_grad_list(paramlist):
        """
        Clears the gradients of all optimized :class:`torch.Tensor` 
        """
        for p in paramlist:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

