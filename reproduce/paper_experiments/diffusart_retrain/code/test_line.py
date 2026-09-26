import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"

import torch
from utils import *
from collections import OrderedDict
# from models.model_atten_unet import Unet
# from models.cond_encoder import cond_encod
from testing.testing import *
# from models.model_coupled import Unet
# from models.model_attent_line_art_v1 import Unet
from models.model_coupled import Unet
from torchvision import transforms
from data.data_load import *
from data.loader_list import *
from training.training_src import *
from torchviz import make_dot
from prettytable import PrettyTable
from torchinfo import summary
import glob




device = "cuda:1" if torch.cuda.is_available() else "cpu"
# load dataset from the hub
image_size = 256
channels = 8 # 3: input noise , 4: nput noise + sketch
batch_size = 1
timesteps= 1000
hint = 'scrib' #Scrib or hint colors
cat = False # Concatenate sketch on input
# original_data, sketchkeras, sketchsimp = init_load_data()
original_data_val, sketchkeras_val, sketchsimp_val = init_load_data_val()

# index = 9
# start = index * 1300
# finish = (index + 1) * 1300
# finish = 13000
# loader_val = MyData_test_scrib(original_data_val, sketchkeras_val, sketchsimp_val, size=image_size)
# loader_val = MyData_eval_scrib(original_data_val[start:finish], sketchkeras_val[start:finish], sketchsimp_val[start:finish], size=image_size)

# loader_val = MyData_eval_scrib(original_data_val, sketchkeras_val, sketchsimp_val, size=image_size)
# print('Starting', start, finish)

original_data_val = glob.glob('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/*.png')
sketchkeras_val = glob.glob('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/sketch/*.jpg')
sketchsimp_val = glob.glob('/home/hec/Phd/Diffusion_v1/paper_test/img_paper/sketch/*.png')
loader_val = MyData_paper_test(original_data_val, sketchkeras_val, sketchsimp_val, size=image_size)

dataloader_test = DataLoader(loader_val, batch_size=batch_size, num_workers=1, shuffle=False)


model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2,)
).to(device)

print('Entering to inference')
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v1/checkpoint_ema_340000.pth')
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v1/checkpoint_ema_390000.pth')
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v2_1px/checkpoint_ema_210000.pth', map_location= device)
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v2_1px_256/checkpoint_ema_129400.pth', map_location= device)
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v2_1px_256_L1/checkpoint_ema_154400.pth', map_location= device)
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v2_1px_256_L1/checkpoint_ema_300000.pth', map_location= device)
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v2_1px_256_L1_CAT/checkpoint_ema_404400.pth', map_location= device)
state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/RGB_hints_scrib_v2_1px_256_L1_CAT_hints_scrib/checkpoint_ema_274400.pth', map_location= device)


new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model.to(device)

if hint == 'scrib':
    # inference_scribs(model, dataloader_test, channels, image_size, device, cat)
    # inference_scribs_evaluation(model, dataloader_test, channels, image_size, device, cat, index)
    # inference_scribs_paper(model, dataloader_test, channels, image_size, device, cat)
    inference_scribs_paper_cat_hints(model, dataloader_test, channels, image_size, device, cat)
elif hint == 'hint':
    inference_hints(model, dataloader_test, channels, image_size, timesteps, device)






# loader_train = MyData_test(original_data, sketchkeras, sketchsimp)
# loader_val = MyData_test(original_data_val, sketchkeras_val, sketchsimp_val)
# loader_val = MyData_train_color(original_data_val, sketchkeras_val, sketchsimp_val, color='lab')
# loader_val = MyData_train_color_flat(original_data_val, sketchkeras, sketchsimp, size= image_size, color='lab')

# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/test_multi_DDP_wo_line/checkpoint.pt')
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/test_line_w_hints_danb_yuv_flat_block_1/checkpoint_ema_150000.pth')
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/test_line_w_hints_danb_v2/checkpoint_ema_400032.pth', map_location= device)
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/test_line_w_hints_danb_v2_LAB_v2_fix/checkpoint_ema_405000.pth')
# state_dict_L = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/test_line_w_hints_danb_L_block_1/checkpoint_ema_225000.pth')
# state_dict = torch.load('/home/hec/Phd/Diffusion_v1/checkpoint/test_line_w_hints_danb_ab_flat_block_1/checkpoint_ema_327200.pth')

# inference_wo_line(model, dataloader_train, channels, image_size, timesteps, device)
# inference_L_hints(model, dataloader_test, channels, image_size, timesteps, device)
# inference_yuv_flat_hints(model, dataloader_test, channels, image_size, timesteps, device)







