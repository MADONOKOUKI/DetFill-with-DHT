import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import torch
from utils import *
from collections import OrderedDict
# from models.model_atten_unet import Unet
# from models.cond_encoder import cond_encod
from testing.testing import *
# from models.model_coupled import Unet
# from models.model_attent_line_art_v1 import Unet
from models.model_line_art_vae import Unet
from torchvision import transforms
from data.data_load import *
from data.loader_list import *
from training.training_src import *
from diffusers import AutoencoderKL
# from torchviz import make_dot
# from prettytable import PrettyTable
# from torchinfo import summary



device = "cuda:0" if torch.cuda.is_available() else "cpu"
# load dataset from the hub
image_size = 64
channels = 8 # 3: input noise , 4: input noise + sketch
batch_size = 1
timesteps= 1000
hint = 'scrib' #Scrib or hint colors
cat = True # Concatenate sketch on input

original_data_val, sketchkeras_val, sketchsimp_val = init_load_data_val()

loader_val = MyData_train_vae(original_data_val, sketchkeras_val, sketchsimp_val, size=512)
dataloader_test = DataLoader(loader_val, batch_size=batch_size, num_workers=1, shuffle=False)

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2,)
).cuda()

print('Entering to inference')

state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_Line_scrib_v2_1px_256_L1_CAT_VAE/checkpoint_ema_25000.pth', map_location= device)

vae = AutoencoderKL().from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae",
                                          torch_dtype=torch.float32).cuda()
for param in vae.parameters():
    param.requires_grad = False

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model.cuda()

inference_scribs_vae(model, dataloader_test, channels, image_size, cat, vae)







