import os
import torch
from utils import *
from collections import OrderedDict
from testing.testing import *
from models.model_coupled_v1 import Unet
from torchvision import transforms
from data.data_load import *
from data.loader_list import *
from data.test_wacv import *
# from training.training_src import *
import glob
import torch.nn as nn

import kornia.color
import numpy as np
import torch
import os
from torchvision.utils import save_image
from pathlib import Path
# from training.forward import *
import torchvision
import PIL
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
from skimage import io
import tqdm

import argparse

parser = argparse.ArgumentParser(description='Diffusion Model Inference script')

# parser.add_argument('--config', default='exp_512/config.yaml')
parser.add_argument('--hint_name', default='dot', type=str, help='path to checkpoint')
parser.add_argument('--checkpoint_path', default='dot', type=str, help='path to checkpoint')
parser.add_argument('--path_length', default=10, type=int, help='path to checkpoint')
parser.add_argument('--ratio', default=0, type=float, help='path to checkpoint')
parser.add_argument('--rnd', default=2, type=int, help='path to checkpoint')
parser.add_argument('--cuda', action='store_false')
parser.add_argument('--domain', default='illust', type=str, help='path to checkpoint')
parser.add_argument('--save_name', default='aaaa', type=str, help='path to checkpoint')
parser.add_argument('--dtype', default='aaaa', type=str, help='path to checkpoint')
parser.add_argument('--dtype_region', default='aaaa', type=str, help='path to checkpoint')

parser.add_argument('--start_idx', default=0, type=int, help='path to checkpoint')
parser.add_argument('--out_root', default='/scratch/madono/det_infer_R3-2', type=str, help='clean output root (replaces hardcoded tttttt)')


args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
# load dataset from the hub
image_size = 256
channels = 8 # 3: input noise , 4: nput noise + sketch
batch_size = 1
timesteps= 1000
hint = 'scrib' #Scrib or hint colors
cat = False # Concatenate sketch on input
device = torch.device("cpu")
device = torch.device("cuda")

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2,)
).to(device)

print('Entering to inference')
# state_dict = torch.load('/home/madorin/gitlab/yuan/hint_colorization/train/compared/Diffusion_v1/checkpoint/wacv_dot/checkpoint_ema_269662.pth', map_location= device)
state_dict = torch.load(args.checkpoint_path, map_location= device)


new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]  # remove `module.`
    new_state_dict[name] = v
# load params
model.load_state_dict(new_state_dict)
model.to(device)


# def init_load_data():

#     #Training data

#     #Training images list
#     with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/train_paper.txt', "r") as a:
#         list_train_imgs = a.readlines()

#     with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/valid_paper.txt', "r") as a:
#         list_valid_imgs = a.readlines()

#     with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/test_paper.txt', "r") as a:
#         list_test_imgs = a.readlines()        

#     return list_train_imgs, list_valid_imgs, list_test_imgs

def init_load_data(domain):

    #Training data

    #Training images list
    with open(os.path.join('configs', str(domain), 'train_paper.txt'), "r") as a:
        list_train_imgs = a.readlines()

    # with open('/home/madorin/gitlab/yuan/hint_colorization/train/config/valid_paper.txt', "r") as a:
    with open(os.path.join('configs', str(domain), 'valid_paper.txt'), "r") as a:

        list_valid_imgs = a.readlines()

    with open(os.path.join('configs', str(domain), 'test_paper.txt'), "r") as a:
        list_test_imgs = a.readlines()        

    return list_train_imgs, list_valid_imgs, list_test_imgs



dataset = "auto" # or annotation
if dataset == "auto":
    list_train_imgs, list_valid_imgs, list_test_imgs = init_load_data(args.domain)

eval_metrics = nn.MSELoss() #.cuda()


hint_ratio = [i * 0.1 for i in range(11)]
# hint_name = 'strokes'
hint_name = args.hint_name


################################################# Forward process #######################################################
timesteps_inf = 1000
timesteps = 1000
# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)
# betas = betas.to(device)
betas = betas
# betas = cosine_beta_schedule(timesteps)
# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
################################################# Forward process #######################################################



def extract_inf(a, t, x_shape):
    batch_size = t.shape[0]
    t= t.to(device)
    a = a.to(device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


@torch.no_grad()
def p_sample_hints_cat(model, x_in, feat, t, t_index, cat):

    noise_pred = model(x_in, feat, t.to(device))

    x = x_in[:, 5::, :, :]

    # print('in', x_in.shape)
    betas_t = extract_inf(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_inf(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract_inf(sqrt_recip_alphas, t, x.shape)
    # print(x.shape, feat.shape)

    # print(t.max(), t.min())
    # print('pred max', noise_pred.max(), 'pred min', noise_pred.min(), 'mean', x.mean(), 'var', x.std())
    # print('proper_pred max', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).max(), 'pred min', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).min(), 'mean', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).mean(), 'var', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).std())
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.to(device)
    betas_t = betas_t.to(device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(device)
    # print(x.shape, noise_pred.shape)
    # noise_pred = torch.cat((feat[:, 0:1, :, :], noise_pred), dim=1)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract_inf(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        noise = noise.to(device)
        sqrt_var = torch.sqrt(posterior_variance_t)
        sqrt_var = sqrt_var.to(device)
        # Algorithm 2 line 4:
        return model_mean + sqrt_var * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop_hints_cat(model, noise, feat, hints, shape, cat=None):
    # device = next(model.parameters()).device
    b = shape[0]
    sketch = feat
    feat = torch.cat((feat[0:b], hints[0:b]), dim=1)
    # start from pure noise (for each example in the batch)

    img = noise[:, 5::, :, :]

    imgs = []

    # for i in tqdm(reversed(range(0, timesteps_inf)), desc='sampling loop time step', total=timesteps_inf):
    for i in tqdm.tqdm(reversed(range(0, timesteps_inf))):
    # for i in reversed(range(0, timesteps_inf)):
        # print(i)
        img = torch.cat((feat[0:b], img[0:b]), dim=1)

        img = p_sample_hints_cat(model, img, feat, torch.full((b,), i, dtype=torch.long), i, cat)
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample_hints_cat(model, noise, feat, hints,image_size, batch_size=16, channels=3, cat=None):
    return p_sample_loop_hints_cat(model, noise, feat, hints, shape=(batch_size, channels, image_size, image_size), cat=cat)


def sample_DD_cat_hints(model, noise, feat_in, hints, scheduler, cat):
    # samples = []
    # noisy_sample = noise[:, 1::, :, :].to(device, dtype=torch.float)
    # sample = noisy_sample.to(device, dtype=torch.float)

    feat = torch.cat((feat_in.to(device, dtype=torch.float), hints.to(device, dtype=torch.float)), dim=1)


    img = noise[:, 5::, :, :].to(device, dtype=torch.float)


    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        t = t.to(device, dtype=torch.float)
        t_tensor = torch.tensor([t, ]).to(device, dtype=torch.float)

        img = torch.cat((feat.to(device, dtype=torch.float), img), dim=1)

            # print(img.shape, feat.shape)

        with torch.no_grad():
            # print(sample.dtype, t.dtype, feat.dtype)
            residual = model(img, feat, t_tensor)


        # 2. compute previous image and set x_t -> x_t-1
        img = scheduler.step(residual, t.long(), img[:, 5::, :, :]).prev_sample




        # samples.append(img)

    return img

# save_path = os.path.join('/home/madorin/gitlab/yuan/hint_colorization/paper_results/compared/Diffusion_v1/', args.hint_name, str(args.path_length), str(args.rnd))
# save_path = os.path.join('/scratch/madono/paper_results_randomhint_diffsuart/paper_results/proposed/Diffusion_v1/', args.hint_name, str(args.path_length), str(args.rnd))

with torch.no_grad():
    def inference_scribs_paper_cat_hints_wacv(save_path, model, dataloader, channels, image_size, device, cat, hint_fix, hint_name, eval_metrics, ratio, batch_size):
        print('in')
        model.eval()
        dir_path = os.path.join(save_path, str(int(ratio * 100)))
        print(dir_path)
        # exit()
        if  os.path.exists(dir_path) is False:
            os.makedirs(dir_path)
        # scheduler = DDPMScheduler(beta_start=1e-4, clip_sample=False)
        # scheduler_DDIM = DDIMScheduler(beta_start=1e-4, clip_sample=False, num_train_timesteps=1000)
        scheduler_DPM = DPMSolverMultistepScheduler(beta_schedule='linear', beta_start=1e-4, algorithm_type='dpmsolver++', solver_order=2, num_train_timesteps=1000, thresholding=True)
        scheduler_PNDM = PNDMScheduler(num_train_timesteps=1000)
        # scheduler.set_timesteps(num_inference_steps=1000)
        # scheduler_DDIM.set_timesteps(num_inference_steps=500)
        scheduler_DPM.set_timesteps(num_inference_steps=100)
        scheduler_PNDM.set_timesteps(num_inference_steps=100)
        for idx_original, batch in enumerate(dataloader):
            # if idx % 1 == 0:
                idx = args.start_idx + idx_original * batch_size
                scheduler_DPM.set_timesteps(num_inference_steps=100)
                print(idx)
                if idx > 4999:
                    break
                if idx < args.start_idx:
                    continue
                if os.path.isfile(os.path.join(dir_path, str(idx) + '_hint.png')):
                    continue            
                batch_size = batch[0].shape[0]
                color = batch[0].to(device).to(dtype=torch.float)
                sketch = batch[1].to(device).to(dtype=torch.float)
                hints = batch[2].to(device).to(dtype=torch.float)

                shape = (batch_size, channels, image_size, image_size)
                torch.manual_seed(2)
                noise = torch.randn(shape, device=device)

                samples_DPM = sample_DD_cat_hints(model, noise, sketch, hints, scheduler_DPM, cat)


                # samples_hints = torchvision.utils.make_grid(reverse_transform_torch(hints[:, 0:3, :, :]))
                # samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                # samples_grid_15 = torchvision.utils.make_grid(reverse_transform_torch(samples_15[-1]))

                samples_DPM = reverse_transform_torch(samples_DPM)
                # samples_DPM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM_15))
                # hints = reverse_transform_torch(hints[:, 0:3, :, :])
                # samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                # samples_grid_15 = torchvision.utils.make_grid(reverse_transform_torch(samples_15[-1]))

                # samples_DPM  = reverse_transform_torch(samples_DPM)
                print('save')

                for j in range(hints.shape[0]):
                    # print(hints.shape)
                    # print(samples_DPM.shape)
                    
                    save_image(reverse_transform_torch(hints[j, 0:3, :, :]).unsqueeze(0), os.path.join(dir_path, str(idx + j) + '_hint.png'))
                    save_image(reverse_transform_torch(color[j, :, :, :]).unsqueeze(0), os.path.join(dir_path, str(idx + j) + '_color.png'))
                    save_image(reverse_transform_torch(sketch[j, :, :, :]).unsqueeze(0), os.path.join(dir_path, str(idx + j) + '_sketch.png'))
                    
                    # save_image(samples_DPM_grid[j, :, :, :].unsqueeze(0), os.path.join(dir_path, str(idx+ j) + '.png'))

                    print(torch.max(samples_DPM),torch.min(samples_DPM))
                    print(torch.max(color),torch.min(color))
                    print(torch.max(sketch),torch.min(sketch))
                    # samples_DPM = (samples_DPM + 1.0) / 2.0
                    save_image(samples_DPM[j, :, :, :].unsqueeze(0), os.path.join(dir_path, str(idx+ j) + '.png'))
                    # exit()                    
                    # save_image(samples_DPM, os.path.join(dir_path, str(idx+ j) + '.png'))
                # exit()




for rnd in range(3):
    # if rnd <=1:
    #     continue
    # save_path = os.path.join('paper_results', args.save_name , args.hint_name, "kernel size_"+str(args.path_length), "hint_"+str(rnd))
    # save_path = os.path.join('/scratch/madono/paper_results_sig25_diffusart/paper_result_userstudy', args.save_name , args.hint_name,  "hint_"+str(rnd))
    save_path = os.path.join('/scratch/madono/paper_results_sig25_diffusart/paper_result_randomhints_2025_rev_sorted', args.save_name , args.hint_name,  "hint_"+str(rnd))
    save_path = os.path.join(args.out_root, args.save_name , args.hint_name,  "hint_"+str(rnd))
    # save_path = os.path.join('/scratch/madono/tog_results/Diffusion_v1_comp/', args.save_name , args.hint_name, "kernel size_"+str(args.path_length), "hint_"+str(rnd))

    # for ratio in hint_ratio:
    img_size, batch_size = 256, 1 #35

    ratio = args.ratio
    hint_fix=True
    valloader = CreateTestLoader(list_test_imgs[args.start_idx:], batch_size, img_size, hint_name, ratio, args.path_length, rnd, args.domain, args.dtype, args.dtype_region)
    # valloader = CreateTrainLoader(list_test_imgs[args.start_idx:], batch_size, img_size, hint_name, ratio, args.path_length, rnd, args.domain, args.dtype, args.dtype_region)


    # pips_avg = inference_paper(model valloader, channels, image_size, device, dataset, hint_name, eval_metrics)
    score_avg = inference_scribs_paper_cat_hints_wacv(save_path, model, valloader, channels, image_size, device, cat, hint_fix, hint_name, eval_metrics, ratio, batch_size)





