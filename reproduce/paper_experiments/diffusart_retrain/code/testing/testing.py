# import imageio
import kornia.color
import numpy as np
import torch
import os
from torchvision.utils import save_image
from pathlib import Path
from utils import *
from training.forward import *
from torch.utils.tensorboard import SummaryWriter
import torchvision
import PIL
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler, PNDMScheduler
from skimage import io
import tqdm

path = '/home/hec/Phd/Diffusion_v1/results/'
# sub_path = 'RGB_hints_scrib_v2_1px_256_L1_CAT/epoch_80/natural/mask_few/'
sub_path = 'RGB_hints_scrib_v2_1px_256_L1_CAT_hints_scrib/epoch_50/'
# sub_path = 'RGB_hints_scrib_v2_1px_256_L1_CAT/keras/test/'

trans = Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1),
    ])


trans_torch = Compose([
    transforms.ToTensor(),
    ])




reverse_transform_f = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: torch.clamp(((t * 255)+0.5), min=0, max=255)),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t.to('cpu', torch.uint8).numpy()),
])

reverse_transform_torch = Compose([
     Lambda(lambda t: (t + 1) / 2),
])
reverse_transform_torch_ab = Compose([
     Lambda(lambda t: (t * 127.0)),
])

def hints_rgb_lab_torch(hints_np):
    hints = torch.unsqueeze(trans_torch(hints_np), 0).to(device)
    hints_lab = kornia.color.rgb_to_lab(hints[:, :3, :, :])
    hints_lab[:, 0, :, :] = hints_lab[:, 0, :, :] / 100.0
    hints_lab[:, 0, :, :] = (hints_lab[:, 0, :, :] * 2) - 1
    hints_lab[:, 1:3, :, :] = hints_lab[:, 1:3, :, :] / 127.0
    # hints[:, 3, :, :] = (hints[:, 3, :, :] * 2) - 1
    hints_lab_a = torch.cat((hints_lab, (hints[:, 2:3, :, :] * 2) - 1), axis=1)

    return hints_lab_a


def torch_rgb_lab(tensor_lab):
    samples_lab = torch.clamp(tensor_lab, min=-1.0, max=1.0)
    samples_lab[:, 0, :, :] = reverse_transform_torch(samples_lab[:, 0, :, :]) * 100.0
    samples_L = samples_lab
    samples_lab[:, 1:, :, :] = reverse_transform_torch_ab(samples_lab[:, 1:, :, :])
    samples_rgb = kornia.color.lab_to_rgb(samples_lab)
    return samples_rgb, samples_L


with torch.no_grad():
    def inference_hints(model, dataloader, channels, image_size, time, device):
        print('in')
        model.eval()
        f =0
        for idx, batch in enumerate(dataloader):

            if idx % 70 == 0:
                batch_size = batch[0].shape[0]
                color = batch[0].to(device)
                sketch = batch[1].to(device)
                hints_np = io.imread('/home/hec/Phd/Diffusion_v1/hints/mask/mask_' + str(idx) + '.png')
                hints = torch.unsqueeze(trans(hints_np), 0).to(device)
                # hints = batch[2].to(device)
                # Write some images in tensorboard every save_train_img step
                # samples = sample(model, sketch_data, image_size=image_size, batch_size=batch_size, channels=channels)
                samples = sample_hints(model, sketch.to(dtype=torch.float), hints.to(dtype=torch.float), 40, image_size=image_size, batch_size=batch_size,
                                 channels=channels)
                samples_15 = sample_hints(model, sketch.to(dtype=torch.float), hints.to(dtype=torch.float), 60, image_size=image_size, batch_size=batch_size, channels=channels)
                samples_25 = sample_hints(model, sketch.to(dtype=torch.float), hints.to(dtype=torch.float), 100, image_size=image_size, batch_size=batch_size,
                                 channels=channels)
                samples_orig_color = torchvision.utils.make_grid(reverse_transform_torch(color))
                samples_orig = torchvision.utils.make_grid(reverse_transform_torch(sketch))
                samples_hints = torchvision.utils.make_grid(reverse_transform_torch(hints))
                samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                samples_grid_10 = torchvision.utils.make_grid(reverse_transform_torch(samples_25[-1]))
                samples_grid_15 = torchvision.utils.make_grid(reverse_transform_torch(samples_15[-1]))

                print('save')
                save_image(samples_orig_color,
                           path + sub_path + 'original_color_' + str(
                               idx) + '.png')

                save_image(samples_orig,
                           path + sub_path + 'original_' + str(
                               idx) + '.png')

                save_image(samples_hints,
                           path + sub_path + 'mask_yes' + str(
                               idx) + '.png')

                save_image(samples_grid,
                           path + sub_path + 'S_1_h' + str(
                               idx) + '.png')

                save_image(samples_grid_15,
                           path + sub_path + 'S15_h' + str(
                               idx) + '.png')
                save_image(samples_grid_10,
                           path + sub_path + 'S25_h' + str(
                               idx) + '.png')
                f += 1
                if f % 7 == 0:
                    break
                # fig = plt.figure()
                # ims = []
                # # sample 64 images
                # for i in range(0,time,5):
                #     im = reverse_transform_f(samples[i][0].cpu())
                #     ims.append(im)
                # imageio.mimsave('/home/hec/Phd/Diffusion_v1/results/test_line_danb_500_EMA_warm_up_2/epoch50/movie_3_new_warn'+str(idx)+'.gif', ims)
                # #

with torch.no_grad():
    def inference_scribs(model, dataloader, channels, image_size, device, cat, hint_fix=True):
        print('in')
        model.eval()
        f =0
        # scheduler = DDPMScheduler(beta_start=1e-4, clip_sample=False)
        # scheduler_DDIM = DDIMScheduler(beta_start=1e-4, clip_sample=False, num_train_timesteps=1000)
        scheduler_DPM = DPMSolverMultistepScheduler(beta_schedule='linear', beta_start=1e-4, algorithm_type='dpmsolver++', solver_order=2, num_train_timesteps=1000, thresholding=True)
        # scheduler_PNDM = PNDMScheduler(num_train_timesteps=1000)
        # scheduler.set_timesteps(num_inference_steps=1000)
        # scheduler_DDIM.set_timesteps(num_inference_steps=500)
        scheduler_DPM.set_timesteps(num_inference_steps=100)
        # scheduler_PNDM.set_timesteps(num_inference_steps=100)
        for idx, batch in enumerate(dataloader):

            if idx % 70 == 0:
                batch_size = batch[0].shape[0]
                color = batch[0].to(device).to(dtype=torch.float)
                sketch = batch[1].to(device).to(dtype=torch.float)

                if hint_fix == True:
                    hints_np = io.imread('/home/hec/Phd/Diffusion_v1/deter_hint/scrib_256/mask_few_' + str(idx) + '.png') / 255.0
                    mask = kornia.utils.image_to_tensor(hints_np[:, :, 0:1])
                    mask = torch.unsqueeze(mask, 0).to(device)
                    hints = torch.cat((color * mask, mask), 1).to(device).to(dtype=torch.float)
                else:
                    hints = batch[2].to(device).to(dtype=torch.float)

                shape = (batch_size, channels, image_size, image_size)
                torch.manual_seed(0)
                noise = torch.randn(shape, device=device)

                if cat ==True:
                    samples = sample_hints(model, noise, sketch, hints, image_size=image_size, batch_size=batch_size,
                                     channels=channels, cat=cat)
                    # samples_DD = sample_DD(model, noise, sketch, hints, batch_size, scheduler)
                    # samples_DDIM = sample_DD(model, noise, sketch, hints, scheduler_DDIM)
                    samples_DPM = sample_DD(model, noise, sketch, hints, scheduler_DPM)
                    samples_PNDM = sample_DD(model, noise, sketch, hints, scheduler_PNDM)

                    torch.manual_seed(5)
                    noise = torch.randn(shape, device=device)
                    samples_15 = sample_hints(model, noise, sketch, hints, image_size=image_size, batch_size=batch_size, channels=channels, cat=cat)
                    # samples_DD_15 = sample_DD(model, noise, sketch, hints, batch_size, scheduler)
                    # samples_DDIM_15 = sample_DD(model, noise, sketch, hints, scheduler_DDIM)
                    samples_DPM_15 = sample_DD(model, noise, sketch, hints, scheduler_DPM)
                    samples_PNDM_15 = sample_DD(model, noise, sketch, hints, scheduler_PNDM)


                    # samples_orig_color = torchvision.utils.make_grid(reverse_transform_torch(color))
                    # samples_orig = torchvision.utils.make_grid(reverse_transform_torch(sketch))
                    samples_hints = torchvision.utils.make_grid(reverse_transform_torch(hints[:, 0:3, :, :]))
                    samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                    samples_grid_15 = torchvision.utils.make_grid(reverse_transform_torch(samples_15[-1]))
                    # samples_grid_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD[-1]))
                    # samples_grid_15_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD_15[-1]))
                    # samples_DDIM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM[-1]))
                    # samples_DDIM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM_15[-1]))
                    samples_PNDM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM[-1]))
                    samples_PNDM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM_15[-1]))
                    samples_DPM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM[-1]))
                    samples_DPM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM_15[-1]))

                print('save')
                # save_image(samples_orig_color,
                #            path + sub_path + 'original_color_' + str(
                #                idx) + '.png')
                # #
                #
                # save_image(samples_orig,
                #            path + sub_path + 'original_' + str(
                #                idx) + '.png')
                # #
                save_image(samples_hints,
                           path + sub_path + 'mask_yes' + str(
                               idx) + '.png')
                #
                #
                save_image(samples_grid,
                           path + sub_path + 'S_1_h' + str(
                               idx) + '.png')

                save_image(samples_grid_15,
                           path + sub_path + 'S15_h' + str(
                               idx) + '.png')


                save_image(samples_DPM_grid,
                           path + sub_path + 'DPM++_2/S_1_h' + str(
                               idx) + '.png')

                save_image(samples_DPM_15_grid,
                           path + sub_path + 'DPM++_2/S15_h' + str(
                               idx) + '.png')
                # sr_input_path = path + sub_path
                # sr_output_path = sr_input_path+"/super"
                # if not os.path.exists(sr_output_path):
                #     os.makedirs(sr_output_path)
                # save_image(samples_DDIM_grid,
                #            path + sub_path + 'DDIM/S_DDIM0' + str(
                #                idx) + '.png')
                # save_image(samples_DDIM_15_grid,
                #            path + sub_path + 'DDIM/S_DDIM15' + str(
                #                idx) + '.png')

                save_image(samples_PNDM_grid,
                           path + sub_path + 'PNDM/S_DDIM0' + str(
                               idx) + '.png')
                save_image(samples_PNDM_15_grid,
                           path + sub_path + 'PNDM/S_DDIM15' + str(
                               idx) + '.png')

                # save_image(samples_DPM_grid,
                #            path + sub_path + 'DPM/S_DPM0' + str(
                #                idx) + '.png')
                # save_image(samples_DPM_15_grid,
                #            path + sub_path + 'DPM/S_DPM15' + str(
                #                idx) + '.png')
                f += 1

                if f % 7 == 0:
                    # os.system(74
                    #     f"cd /home/hec/Phd/other_colorization/nunif; python -m waifu2x.cli -m scale  -i  {sr_input_path} -o {sr_output_path}")

                    break
                # fig = plt.figure()
                # ims = []
                # # sample 64 images
                # for i in range(0,time,5):
                #     im = reverse_transform_f(samples[i][0].cpu())
                #     ims.append(im)
                # imageio.mimsave('/home/hec/Phd/Diffusion_v1/results/test_line_danb_500_EMA_warm_up_2/epoch50/movie_3_new_warn'+str(idx)+'.gif', ims)

with torch.no_grad():
    def inference_scribs_vae(model, dataloader, channels, image_size, cat, vae, hint_fix=True):
        print('in')
        model.eval()
        f =0
        # scheduler = DDPMScheduler(beta_start=1e-4, clip_sample=False)
        # scheduler_DDIM = DDIMScheduler(beta_start=1e-4, clip_sample=False)
        # scheduler_DPM = DPMSolverMultistepScheduler(beta_schedule='linear', algorithm_type='dpmsolver', solver_order=3)
        # scheduler_PNDM = PNDMScheduler()
        # scheduler.set_timesteps(num_inference_steps=1000)
        # scheduler_DDIM.set_timesteps(num_inference_steps=500)
        # scheduler_DPM.set_timesteps(num_inference_steps=250)
        # scheduler_PNDM.set_timesteps(num_inference_steps=250)
        for idx, batch in enumerate(dataloader):

            if idx % 70 == 0:
                batch_size = batch[0].shape[0]
                # color = batch[0].to(dtype=torch.float).cuda()
                sketch = batch[1].to(dtype=torch.float).cuda()

                latent_sketch = pil_to_latents(sketch, vae)

                # if hint_fix == True:
                #     hints_np = io.imread('/home/hec/Phd/Diffusion_v1/deter_hint/scrib_256/mask_' + str(idx) + '.png') / 255.0
                #     mask = kornia.utils.image_to_tensor(hints_np[:, :, 0:1])
                #     mask = torch.unsqueeze(mask, 0).to(device)
                #     hints = torch.cat((color * mask, mask), 1).to(device).to(dtype=torch.float)
                # else:
                #     hints = batch[2].to(device).to(dtype=torch.float)

                shape = (batch_size, 4, 64, 64)
                torch.manual_seed(0)
                noise = torch.randn(shape).cuda()

                if cat ==True:
                    samples = sample_DDPM(model, noise, latent_sketch)
                    sample_img = latents_to_pil(samples.cuda(), vae)

                    torch.manual_seed(5)
                    noise = torch.randn(shape).cuda()
                    samples_15 = sample_DDPM(model, noise, latent_sketch)
                    sample_img_15 = latents_to_pil(samples_15.cuda(), vae)


                    # samples_orig_color = torchvision.utils.make_grid(reverse_transform_torch(color))
                    # samples_orig = torchvision.utils.make_grid(reverse_transform_torch(sketch))
                    # samples_hints = torchvision.utils.make_grid(reverse_transform_torch(hints[:, 0:3, :, :]))
                    samples_grid = torchvision.utils.make_grid(sample_img)
                    samples_grid_15 = torchvision.utils.make_grid(sample_img_15)
                    # samples_grid_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD[-1]))
                    # samples_grid_15_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD_15[-1]))
                    # samples_DDIM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM[-1]))
                    # samples_DDIM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM_15[-1]))
                    samples_PNDM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM[-1]))
                    samples_PNDM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM_15[-1]))
                    samples_DPM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM[-1]))
                    samples_DPM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM_15[-1]))


                print('save')
                # save_image(samples_orig_color,
                #            path + sub_path + 'original_color_' + str(
                #                idx) + '.png')
                #
                #
                # save_image(samples_orig,
                #            path + sub_path + 'original_' + str(
                #                idx) + '.png')

                # save_image(samples_hints,
                #            path + sub_path + 'mask_yes' + str(
                #                idx) + '.png')


                save_image(samples_grid,
                           path + sub_path + 'S_1_h' + str(
                               idx) + '.png')

                save_image(samples_grid_15,
                           path + sub_path + 'S15_h' + str(
                               idx) + '.png')

                #
                save_image(samples_grid_DD,
                           path + sub_path + 'DPM++_3/S_1_h' + str(
                               idx) + '.png')

                save_image(samples_grid_15_DD,
                           path + sub_path + 'DPM++_3/S15_h' + str(
                               idx) + '.png')
                # sr_input_path = path + sub_path
                # sr_output_path = sr_input_path+"/super"
                # if not os.path.exists(sr_output_path):
                #     os.makedirs(sr_output_path)
                # save_image(samples_DDIM_grid,
                #            path + sub_path + 'DDIM/S_DDIM0' + str(
                #                idx) + '.png')
                # save_image(samples_DDIM_15_grid,
                #            path + sub_path + 'DDIM/S_DDIM15' + str(
                #                idx) + '.png')

                # save_image(samples_PNDM_grid,
                #            path + sub_path + 'PNDM/S_DDIM0' + str(
                #                idx) + '.png')
                # save_image(samples_PNDM_15_grid,
                #            path + sub_path + 'PNDM/S_DDIM15' + str(
                #                idx) + '.png')
                #
                # save_image(samples_DPM_grid,
                #            path + sub_path + 'DPM/S_DPM0' + str(
                #                idx) + '.png')
                # save_image(samples_DPM_15_grid,
                #            path + sub_path + 'DPM/S_DPM15' + str(
                #                idx) + '.png')
                f += 1

                if f % 7 == 0:
                    # os.system(
                    #     f"cd /home/hec/Phd/other_colorization/nunif; python -m waifu2x.cli -m scale  -i  {sr_input_path} -o {sr_output_path}")

                    break
                # fig = plt.figure()
                # ims = []
                # # sample 64 images
                # for i in range(0,time,5):
                #     im = reverse_transform_f(samples[i][0].cpu())
                #     ims.append(im)
                # imageio.mimsave('/home/hec/Phd/Diffusion_v1/results/test_line_danb_500_EMA_warm_up_2/epoch50/movie_3_new_warn'+str(idx)+'.gif', ims)


with torch.no_grad():
    def inference_scribs_evaluation(model, dataloader, channels, image_size, device, cat, index):
        print('in')
        model.eval()
        f =0
        batch_size = 1
        scheduler = DDPMScheduler(beta_start=1e-4, clip_sample=False)
        scheduler.set_timesteps(num_inference_steps=1000)
        # scheduler_DPM = DPMSolverMultistepScheduler(beta_schedule='linear', beta_start=1e-4,
        #                                             algorithm_type='dpmsolver++', solver_order=2,
        #                                             num_train_timesteps=1000, thresholding=True)
        # scheduler_DPM.set_timesteps(num_inference_steps=100)
        # scheduler_PNDM = PNDMScheduler(num_train_timesteps=1000)
        # scheduler_PNDM.set_timesteps(num_inference_steps=100)
        shape = (batch_size, channels, image_size, image_size)
        for idx, batch in enumerate(dataloader):
            print((index*1300)+idx, '/', (len(dataloader)*((index +1))))
            # color = batch[0].to(device).to(dtype=torch.float)
            sketch = batch[1].to(device).to(dtype=torch.float)
            hints = batch[2].to(device).to(dtype=torch.float)
            name = batch[3]


            torch.manual_seed(5)
            noise = torch.randn(shape, device=device)
            samples_PNDM = sample_DD(model, noise, sketch, hints, scheduler, cat)
            samples_PNDM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM))

            save_image(samples_PNDM_grid,
                         path + sub_path + name[0])

with torch.no_grad():
    def inference_scribs_paper(model, dataloader, channels, image_size, device, cat, hint_fix=True):
        print('in')
        model.eval()
        # scheduler = DDPMScheduler(beta_start=1e-4, clip_sample=False)
        # scheduler_DDIM = DDIMScheduler(beta_start=1e-4, clip_sample=False, num_train_timesteps=1000)
        scheduler_DPM = DPMSolverMultistepScheduler(beta_schedule='linear', beta_start=1e-4, algorithm_type='dpmsolver++', solver_order=2, num_train_timesteps=1000, thresholding=True)
        scheduler_PNDM = PNDMScheduler(num_train_timesteps=1000)
        # scheduler.set_timesteps(num_inference_steps=1000)
        # scheduler_DDIM.set_timesteps(num_inference_steps=500)
        scheduler_DPM.set_timesteps(num_inference_steps=100)
        scheduler_PNDM.set_timesteps(num_inference_steps=100)
        for idx, batch in enumerate(dataloader):
            if idx % 1 == 0:
                batch_size = batch[0].shape[0]
                color = batch[0].to(device).to(dtype=torch.float)
                sketch = batch[1].to(device).to(dtype=torch.float)
                hints = batch[2].to(device).to(dtype=torch.float)

                shape = (batch_size, channels, image_size, image_size)
                torch.manual_seed(2)
                noise = torch.randn(shape, device=device)

                samples = sample_hints(model, noise, sketch, hints, image_size=image_size, batch_size=batch_size,
                                         channels=channels, cat=cat)
                # samples_DD = sample_DD(model, noise, sketch, hints, batch_size, scheduler)
                samples_DPM = sample_DD(model, noise, sketch, hints, scheduler_DPM, cat)
                samples_PNDM = sample_DD(model, noise, sketch, hints, scheduler_PNDM, cat)

                torch.manual_seed(5)
                noise = torch.randn(shape, device=device)
                samples_15 = sample_hints(model, noise, sketch, hints, image_size=image_size, batch_size=batch_size, channels=channels, cat=cat)
                # samples_DD_15 = sample_DD(model, noise, sketch, hints, batch_size, scheduler)


                samples_DPM_15 = sample_DD(model, noise, sketch, hints, scheduler_DPM, cat)
                samples_PNDM_15 = sample_DD(model, noise, sketch, hints, scheduler_PNDM, cat)


                # samples_orig_color = torchvision.utils.make_grid(reverse_transform_torch(color))
                # samples_orig = torchvision.utils.make_grid(reverse_transform_torch(sketch))
                samples_hints = torchvision.utils.make_grid(reverse_transform_torch(hints[:, 0:3, :, :]))
                samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                samples_grid_15 = torchvision.utils.make_grid(reverse_transform_torch(samples_15[-1]))
                # samples_grid_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD[-1]))
                # samples_grid_15_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD_15[-1]))
                # samples_DDIM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM[-1]))
                # samples_DDIM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM_15[-1]))
                samples_PNDM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM))
                samples_PNDM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM_15))
                samples_DPM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM))
                samples_DPM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM_15))

                print('save')

                save_image(samples_hints,
                            path + sub_path + 'mask' + str(
                                idx) + '.png')

                save_image(samples_grid,
                               path + sub_path + 'S_1_h' + str(
                                   idx) + '.png')

                save_image(samples_grid_15,
                               path + sub_path + 'S15_h' + str(
                                   idx) + '.png')


                save_image(samples_DPM_grid,
                               path + sub_path + 'DPM++_2/S_1_h' + str(
                                   idx) + '.png')

                save_image(samples_DPM_15_grid,
                               path + sub_path + 'DPM++_2/S15_h' + str(
                                   idx) + '.png')

                save_image(samples_PNDM_grid,
                               path + sub_path + 'PNDM/S_DDIM0' + str(
                                   idx) + '.png')
                save_image(samples_PNDM_15_grid,
                               path + sub_path + 'PNDM/S_DDIM15' + str(
                                   idx) + '.png')

with torch.no_grad():
    def inference_scribs_paper_cat_hints(model, dataloader, channels, image_size, device, cat, hint_fix=True):
        print('in')
        model.eval()
        # scheduler = DDPMScheduler(beta_start=1e-4, clip_sample=False)
        # scheduler_DDIM = DDIMScheduler(beta_start=1e-4, clip_sample=False, num_train_timesteps=1000)
        scheduler_DPM = DPMSolverMultistepScheduler(beta_schedule='linear', beta_start=1e-4, algorithm_type='dpmsolver++', solver_order=2, num_train_timesteps=1000, thresholding=True)
        scheduler_PNDM = PNDMScheduler(num_train_timesteps=1000)
        # scheduler.set_timesteps(num_inference_steps=1000)
        # scheduler_DDIM.set_timesteps(num_inference_steps=500)
        scheduler_DPM.set_timesteps(num_inference_steps=100)
        scheduler_PNDM.set_timesteps(num_inference_steps=100)
        for idx, batch in enumerate(dataloader):
            if idx % 1 == 0:
                batch_size = batch[0].shape[0]
                color = batch[0].to(device).to(dtype=torch.float)
                sketch = batch[1].to(device).to(dtype=torch.float)
                hints = batch[2].to(device).to(dtype=torch.float)

                shape = (batch_size, channels, image_size, image_size)
                torch.manual_seed(2)
                noise = torch.randn(shape, device=device)

                samples = sample_hints_cat(model, noise, sketch, hints, image_size=image_size, batch_size=batch_size,
                                         channels=channels, cat=cat)
                # samples_DD = sample_DD(model, noise, sketch, hints, batch_size, scheduler)
                samples_DPM = sample_DD_cat_hints(model, noise, sketch, hints, scheduler_DPM, cat)
                samples_PNDM = sample_DD_cat_hints(model, noise, sketch, hints, scheduler_PNDM, cat)

                torch.manual_seed(5)
                noise = torch.randn(shape, device=device)
                samples_15 = sample_hints_cat(model, noise, sketch, hints, image_size=image_size, batch_size=batch_size, channels=channels, cat=cat)
                # samples_DD_15 = sample_DD(model, noise, sketch, hints, batch_size, scheduler)


                samples_DPM_15 = sample_DD_cat_hints(model, noise, sketch, hints, scheduler_DPM, cat)
                samples_PNDM_15 = sample_DD_cat_hints(model, noise, sketch, hints, scheduler_PNDM, cat)


                # samples_orig_color = torchvision.utils.make_grid(reverse_transform_torch(color))
                # samples_orig = torchvision.utils.make_grid(reverse_transform_torch(sketch))
                samples_hints = torchvision.utils.make_grid(reverse_transform_torch(hints[:, 0:3, :, :]))
                samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                samples_grid_15 = torchvision.utils.make_grid(reverse_transform_torch(samples_15[-1]))
                # samples_grid_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD[-1]))
                # samples_grid_15_DD = torchvision.utils.make_grid(reverse_transform_torch(samples_DD_15[-1]))
                # samples_DDIM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM[-1]))
                # samples_DDIM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DDIM_15[-1]))
                samples_PNDM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM))
                samples_PNDM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_PNDM_15))
                samples_DPM_grid  = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM))
                samples_DPM_15_grid = torchvision.utils.make_grid(reverse_transform_torch(samples_DPM_15))

                print('save')

                save_image(samples_hints,
                            path + sub_path + 'mask' + str(
                                idx) + '.png')

                save_image(samples_grid,
                               path + sub_path + 'S_1_h' + str(
                                   idx) + '.png')

                save_image(samples_grid_15,
                               path + sub_path + 'S15_h' + str(
                                   idx) + '.png')


                save_image(samples_DPM_grid,
                               path + sub_path + 'DPM++_2/S_1_h' + str(
                                   idx) + '.png')

                save_image(samples_DPM_15_grid,
                               path + sub_path + 'DPM++_2/S15_h' + str(
                                   idx) + '.png')

                save_image(samples_PNDM_grid,
                               path + sub_path + 'PNDM/S_DDIM0' + str(
                                   idx) + '.png')
                save_image(samples_PNDM_15_grid,
                               path + sub_path + 'PNDM/S_DDIM15' + str(
                                   idx) + '.png')