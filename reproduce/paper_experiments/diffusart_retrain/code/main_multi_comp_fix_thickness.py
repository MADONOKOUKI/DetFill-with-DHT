import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7,8,9"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

import torch
import torchvision
from torchvision import transforms
from copy import deepcopy
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from utils import *
# from models.model_attent_line_art import Unet
from models.model_coupled import Unet
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler
# from .training.training_multi import *
from training import *
from data.data_load import *
from data.loader_list import *

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# device = "cuda:5" if torch.cuda.is_available() else "cpu"

def init_distributed():

    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()
    setup_for_distributed(rank == 0)

if __name__ == '__main__':
    # load dataset from the hub
    print("Cuda support:", torch.cuda.is_available(),":", torch.cuda.device_count(), "devices")
    init_distributed()
    image_size = 256
    channels = 8
    batch_size = 2
    timesteps = 1000

    ########################################################


    ################### Loading data ####################
    list_train_imgs, list_valid_imgs = init_load_data('illust')
    loader_train = MyData_train_scrib(list_train_imgs, size=image_size, approach='comp', hint='strokes', domain='illust', dtype='deep_illust_pretrained', dtype_region='segmentation', fix_thickness=4)

    # loader_val = MyData_train(original_data, sketchkeras, sketchsimp)

    # Distributed sampler
    sampler = DistributedSampler(loader_train, shuffle=True, drop_last=True)

    # create dataloader
    dataloader_train = DataLoader(loader_train, batch_size=batch_size, num_workers=10, shuffle=False, sampler=sampler)
    # dataloader_train = torch.utils.data.DataLoader(loader_train , batch_size=batch_size, num_workers=5,
    #                                           shuffle=True)
    ############## Defining model #####################"
    local_rank = int(os.environ['LOCAL_RANK'])

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2,)
    )
    model.cuda()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP((model), device_ids=[local_rank], find_unused_parameters=True)
    model_ema = deepcopy(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    sub_dir = "tog2024_scribble"
    training_dif_multi_hints(model, model_ema, dataloader_train, optimizer, timesteps, channels, image_size, sub_dir )




