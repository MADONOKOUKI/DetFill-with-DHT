import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1,3"


import torch
import torchvision
from torchvision import transforms
from copy import deepcopy
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from utils import *
from models.model_attent_line_art import Unet
from torch.optim import Adam
from torch.utils.data.distributed import DistributedSampler
from data.data_load import *
from data.loader_list import *
from training.training_multi import *
from torchviz import make_dot
from prettytable import PrettyTable
from torchinfo import summary



import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    # load dataset from the hub

    image_size = 256
    channels = 3
    batch_size = 4
    timesteps = 1000

    ########################################################


    ################### Loading data ####################
    original_data, sketchkeras, sketchsimp = init_load_data()
    # original_data_val, sketchkeras_val, sketchsimp_val = init_load_data_val()

    loader_train = MyData_train(original_data, sketchkeras, sketchsimp)
    # loader_val = MyData_train(original_data, sketchkeras, sketchsimp)


    # create dataloader
    dataloader_train = DataLoader(loader_train, batch_size=batch_size, num_workers=5, shuffle=False)

    # dataloader_train = torch.utils.data.DataLoader(loader_train , batch_size=batch_size, num_workers=5,
    #                                           shuffle=True)
    ############## Defining model #####################"


    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)
    model_ema = deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    training_dif_hints(model, model_ema, dataloader_train, optimizer, timesteps, channels, image_size, device)





