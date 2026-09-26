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
# from training.training_multi import training_dif_multi_wo_line
from torchviz import make_dot
from prettytable import PrettyTable
from torchinfo import summary



import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

device = "cuda:2" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    # load dataset from the hub
    # init_distributed()
    image_size = 128
    channels = 3
    batch_size = 2
    timesteps = 1000

    ########################################################


    ################### Loading data ####################
    original_data, sketchkeras, sketchsimp = init_load_data()
    # original_data_val, sketchkeras_val, sketchsimp_val = init_load_data_val()

    loader_train = MyData_train(original_data, sketchkeras, sketchsimp)
    # loader_val = MyData_train(original_data, sketchkeras, sketchsimp)

    # Distributed sampler
    # sampler = DistributedSampler(loader_train, shuffle=True, drop_last=False)

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
    model = model.to(device)

    x = torch.zeros((2, 3, 128, 128)).to(device)
    s = torch.zeros((2,1, 128, 128)).to(device)
    t = torch.randint(0, timesteps, (batch_size,), device=None).long().cuda().to(device)

    summary(model, input_data=[x, s, t])

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = DDP((model), device_ids=[local_rank], find_unused_parameters=True)
    # model_ema = deepcopy(model)






