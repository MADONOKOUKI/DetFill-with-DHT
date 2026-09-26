import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import torchvision
from torchvision import transforms
from copy import deepcopy
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from utils import *
# from models.model_attent_line_art import Unet
# from models.model_coupled import Unet
from models.model_line_art_vae import Unet
from torch.optim import Adam
from data.data_load import *
from data.loader_list import *
from training.training_multi import training_dif_multi_hints_vae
from diffusers import AutoencoderKL





# device = "cuda:2" if torch.cuda.is_available() else "cpu"



if __name__ == '__main__':
    # load dataset from the hub
    image_size = 64
    channels = 8
    batch_size = 4
    timesteps = 1000

    ########################################################


    ################### Loading data ####################
    original_data, sketchkeras, sketchsimp = init_load_data()
    original_data_val, sketchkeras_val, sketchsimp_val = init_load_data_val()

    # loader_train = MyData_train_scrib(original_data, sketchkeras, sketchsimp, size=image_size)
    loader_val = MyData_train_vae(original_data_val[200:208], sketchkeras_val[200:208], sketchsimp_val[200:208], size=512)
    loader_train = MyData_train_vae(original_data, sketchkeras, sketchsimp, size=512)


    # create dataloader
    dataloader_train = DataLoader(loader_train, batch_size=batch_size, num_workers=4, shuffle=True)
    dataloader_val = DataLoader(loader_val, batch_size=batch_size, num_workers=4, shuffle=False)

    ############## Defining model #####################"

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2,)
    )
    model.cuda()

    vae = AutoencoderKL().from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                          torch_dtype=torch.float32).cuda()
    for param in vae.parameters():
        param.requires_grad = False

    model_ema = deepcopy(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # training_dif_hints(model, model_ema, dataloader_train, optimizer, timesteps, channels, image_size)
    training_dif_multi_hints_vae(model, model_ema, vae, dataloader_train, dataloader_val, optimizer, timesteps, channels, image_size)


