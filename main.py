# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from absl import app
from absl import flags
import cv2
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import pdb
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from nnutils.train_utils import v2s_trainer

import os

opts = flags.FLAGS
def main(_):
    # This prints out all the flags to the console.
    for flag in opts.flag_values_dict():
        print(f"{flag}: {opts[flag].value}")
    # Diagnostic Information
    print("os.environ.get:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print("PyTorch version:", torch.__version__)
    import torchvision
    print("TorchVision version:", torchvision.__version__)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.device_count():", torch.cuda.device_count())
    print("opts.local_rank:", opts.local_rank)
    print("MASTER_ADDR:", os.environ.get('MASTER_ADDR', 'Not Set'))
    print("MASTER_PORT:", os.environ.get('MASTER_PORT', 'Not Set'))
    
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(opts.local_rank))

    try:
        torch.cuda.set_device(opts.local_rank)
        world_size = opts.ngpu
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=world_size,
            rank=opts.local_rank,
        )
        #print('%d/%d' % (world_size, opts.local_rank))
        print('world_size: %d  opts.local_rank: %d' % (world_size, opts.local_rank))
    except Exception as e:
        print("Error while initializing CUDA or distributed process group:", str(e))
        return

    torch.manual_seed(0)
    torch.cuda.manual_seed(1)
    torch.manual_seed(0)
    
    trainer = v2s_trainer(opts)
    data_info = trainer.init_dataset()    
    trainer.define_model(data_info)
    trainer.init_training()
    trainer.train()

if __name__ == '__main__':
    app.run(main)
