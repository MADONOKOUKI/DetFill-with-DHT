# DDP_FIX_V1 applied (DATESTAMP)
import torch
import torch.distributed as dist
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import optimizer, Adam
import torchvision
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision.utils import save_image
from pathlib import Path
from utils import *
from training.losses import *
from training.forward import *
from torch.utils.tensorboard import SummaryWriter
import time
import imageio
from diffusers.optimization import get_cosine_schedule_with_warmup
import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from diffusers import DPMSolverMultistepScheduler
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

with torch.no_grad():
    def sample_DD(model, feat, batch_size, scheduler):
        samples = []
        seed = 5
        torch.manual_seed(seed)
        noisy_sample = torch.randn(
            batch_size, 1, 256, 256
        )
        sample = noisy_sample.cuda()
        for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
            # 1. predict noise residual
            t = t.cuda()
            t_tensor = torch.tensor([t, ]).cuda()
            with torch.no_grad():
                # print(sample.dtype, t.dtype, feat.dtype)
                residual = model(sample.to(dtype=torch.float), feat.to(dtype=torch.float), t_tensor.to(dtype=torch.float))

            # 2. compute previous image and set x_t -> x_t-1
            sample = scheduler.step(residual, t.long(), sample).prev_sample

            samples.append(sample)

        return samples

with torch.no_grad():
    def inference_hints_DDIM_in_training(model, feat, batch_size):
        print('in')
        model.eval()
        scheduler = DPMSolverMultistepScheduler(beta_schedule='linear', algorithm_type='dpmsolver', )
        scheduler.set_timesteps(num_inference_steps=25)
        samples = sample_DD(model, feat, batch_size, scheduler)

        return samples

# Checkpoint dir
# sub_dir = 'RGB_Line_scrib_v2_1px_256_L1_CAT_VAE_V2'
# checkpoint_dir = ('./checkpoint/' + sub_dir)
# full_checkpoint_dir = ('./checkpoint/' + sub_dir +'/checkpoint.pth')
# full_checkpoint_dir_ema = ('./checkpoint/' + sub_dir +'/checkpoint_ema.pth')
# model_dir = ('./save_models/' + sub_dir)
PATH = './net_start.pth'
# Save images in tensorboard
# save_val_img = 200
# save_train_img = 50
save_model_train = 2200
ema_decay = 0.995
lr_warmup_steps = 5000 #2000
save_img = 2000

# checkpoint_dir = ('./checkpoint/' + sub_dir)
# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir, exist_ok=True)

epochs = 200


loss_val_epoch = 0.0
running_loss = 0.0

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
])

reverse_transform_torch = Compose([
     Lambda(lambda t: (t + 1) / 2),
])

reverse_transform_torch_ab = Compose([
     Lambda(lambda t: (t * 127)),
])

def training_dif_multi_hints_vae(model, model_ema, vae, dataloader, dataloader_val, optimizer, timesteps, channels, image_size):
    val_iter = int(len(dataloader)/ 2)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(dataloader) * epochs),
    )

    # scaler = torch.cuda.amp.GradScaler()

    # Writer will output to ./runs/ directory by default
    tb = SummaryWriter('./runs/' + sub_dir + '/', flush_secs=1)
    model.train()
    for epoch in range(epochs):
        model.train()
        dataloader.sampler.set_epoch(epoch)
        dataloader_val.sampler.set_epoch(epoch)
        if epoch == 0:
            sum_idx = 0
            sum_idx_val =0
            running_loss = 0.0
        else:
            sum_idx += len(dataloader)
            sum_idx_val += len(dataloader_val)
            running_loss = 0.0

        for idx, batch in enumerate(dataloader):
            step_start_time = time.time()
            optimizer.zero_grad()

            batch_size = batch[0].shape[0]
            original = batch[0].to(dtype=torch.float32).cuda()
            sketch = batch[1].to(dtype=torch.float32).cuda()


            latent_gt = pil_to_latents(original, vae)
            latent_sketch = pil_to_latents(sketch, vae)



            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=None).long().cuda()

            loss = p_losses_lines(model, latent_gt, latent_sketch, t, loss_type="l1")


            if idx % 50 == 0:
                print(
                    "Epoch[{}/{}]({}/{}): Loss_train: {:.4f}, lr_step: {:.4f}, Step time: {} seconds".format(
                        epoch, epochs, idx, len(dataloader),
                        loss.item(), lr_scheduler.get_last_lr()[0], time.time() - step_start_time))

            running_loss = running_loss + loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if is_main_process:
                ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)

                # Write some images in tensorboard every save_train_img step
                if (epoch % 2 == 0) and (idx % save_model_train == 0):
                    torch.save(model.state_dict(), './checkpoint/' + sub_dir +'/checkpoint_'+str(idx + sum_idx)+'.pth')
                    torch.save(model_ema.state_dict(), './checkpoint/' + sub_dir +'/checkpoint_ema_'+str(idx + sum_idx)+'.pth')

                    # save model checkpoint
                    torch.save({
                        'epoch': epochs,
                        'model_state_dict': model.state_dict(),
                        'model_state_dict_ema': model_ema.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, './checkpoint/' + sub_dir +'/checkpoint_resumen.pth')
                #

                # Writing losses in tensorboard
                tb.add_scalar("Total loss per step", running_loss / (idx + 1), idx + sum_idx)


                tb.flush()
            if idx % val_iter==0:
                shape_val = (batch_size, 4, 64, 64)
                with torch.no_grad():
                    for idx_val, batch_val in enumerate(dataloader_val):
                        # batch_size = batch_val[0].shape[0]
                        original = batch_val[0].to(dtype=torch.float32).cuda()
                        sketch = batch_val[1].to(dtype=torch.float32).cuda()
                        latent_sketch = pil_to_latents(sketch, vae)
                        torch.manual_seed(0)
                        noise_val = torch.randn(shape_val).cuda()
                        # samples = sample_hints_vae(model, noise_val, latent_sketch, image_size=image_size, batch_size=batch_size,
                        #                  channels=channels, cat=True)
                        if is_main_process:
                            samples = sample_DDPM(model, noise_val, latent_sketch)
                            sample_img = latents_to_pil(samples.cuda(), vae)
                            samples_orig = torchvision.utils.make_grid(reverse_transform_torch(original))
                            samples_sketch = torchvision.utils.make_grid(reverse_transform_torch(sketch))
                            # samples_hint = torchvision.utils.make_grid((reverse_transform_torch(hints[:,0:3,:, :])))
                            samples_grid = torchvision.utils.make_grid(sample_img)
                            tb.add_image('Color in step', samples_orig, idx_val + sum_idx)
                            tb.add_image('Sketch in step', samples_sketch, idx_val + sum_idx)
                            # tb.add_image('Hint mask in step', samples_hint, idx + sum_idx)
                            tb.add_image('Sample in step', samples_grid, idx_val + sum_idx)

        if is_main_process:
            # Average loss for each epoch
            epoch_loss = running_loss / len(dataloader)

            tb.add_scalar("Total loss per epoch", epoch_loss, epoch)

        # Save final model
    if is_main_process:
        torch.save(model.state_dict(), full_checkpoint_dir)
        torch.save(model_ema.state_dict(), full_checkpoint_dir_ema)
        print('Training saved and finished')
        tb.close()

def training_dif_multi_hints(model, model_ema, dataloader, optimizer, timesteps, channels, image_size, sub_dir, accumulation_steps=1):


    checkpoint_dir = ('./checkpoint/' + sub_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)


    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * accumulation_steps,
        num_training_steps=(len(dataloader) * epochs),
    )

    # scaler = torch.cuda.amp.GradScaler()

    # Writer will output to ./runs/ directory by default
    class _NullSink:
        def __getattr__(self, _):
            return lambda *a, **k: None
    if is_main_process():
        tb = SummaryWriter('./runs/' + sub_dir + '/', flush_secs=10)
    else:
        tb = _NullSink()
    def _barrier():
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    model.train()
    for epoch in range(epochs):
        model.train()
        dataloader.sampler.set_epoch(epoch)
        if epoch == 0:
            best_loss = 1000.0
            sum_idx = 0
            running_loss = 0.0
        else:
            sum_idx += len(dataloader)
            running_loss = 0.0

        for idx, batch in enumerate(dataloader):
            # torch.cuda.empty_cache()
            # print(idx)
            step_start_time = time.time()

            batch_size = batch[0].shape[0]
            original = batch[0].to(dtype=torch.float).cuda()
            sketch = batch[1].to(dtype=torch.float).cuda()
            # original_L = batch[2].cuda()
            hints = batch[2].to(dtype=torch.float).cuda()
            # mask = hints[:, 3:4, :, :]
            # hints_L = hints[:, 0:1, :, :]
            # hints_L = torch.cat((hints_L, mask), dim=1)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (original.shape[0],), device=None).long().cuda()

            loss = p_losses_hints(model, original, sketch, hints, t, loss_type="l1")
            # with torch.cuda.amp.autocast():
            #     loss = p_losses_hints(model, original, sketch, hints, t, loss_type="huber")

            if idx % 50 == 0:
                print(
                    "Epoch[{}/{}]({}/{}): Loss_train: {:.4f}, lr_step: {:.4f}, Step time: {} seconds".format(
                        epoch, epochs, idx, len(dataloader),
                        loss.item(), lr_scheduler.get_last_lr()[0], time.time() - step_start_time))

            running_loss = running_loss + loss.cpu().item()

            loss.backward()
            if idx % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            lr_scheduler.step()
            pass  # torch.cuda.empty_cache() removed: numerics-neutral, dominated step time on 49GB ctx

            # continue

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # lr_scheduler.step()

            if idx % accumulation_steps == 0:
                ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
            if is_main_process():

                # Write some images in tensorboard every save_train_img step
                if (epoch % 2 == 0) and (idx % save_model_train == 0):
                    torch.save(model.state_dict(), './checkpoint/' + sub_dir +'/checkpoint_'+str((idx + sum_idx)//accumulation_steps)+'.pth')
                    torch.save(model_ema.state_dict(), './checkpoint/' + sub_dir +'/checkpoint_ema_'+str((idx + sum_idx)//accumulation_steps)+'.pth')

                    # save model checkpoint
                    torch.save({
                        'epoch': epoch,
                        'global_step': idx + sum_idx,
                        'model_state_dict': model.state_dict(),
                        'model_state_dict_ema': model_ema.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, './checkpoint/' + sub_dir +'/checkpoint_resumen.pth')
                #
                if (idx % save_img==0):
                    # samples = inference_hints_DDIM_in_training(model_ema, feat, batch_size)
                    samples_orig = torchvision.utils.make_grid(reverse_transform_torch(original.cpu()))
                    samples_sketch = torchvision.utils.make_grid(reverse_transform_torch(sketch.cpu()))
                    samples_hint = torchvision.utils.make_grid((reverse_transform_torch(hints[:,0:3,:, :].cpu())))
                    # samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                    tb.add_image('Color in step', samples_orig, idx + sum_idx)
                    tb.add_image('Sketch in step', samples_sketch, idx + sum_idx)
                    tb.add_image('Hint mask in step', samples_hint, idx + sum_idx)
                    # tb.add_image('Sample in step', samples_grid, idx + sum_idx)

                # Writing losses in tensorboard
                tb.add_scalar("Total loss per step", running_loss / (idx + 1), idx + sum_idx)
            _barrier()


        if is_main_process():
            # Average loss for each epoch
            epoch_loss = running_loss / len(dataloader)

            tb.add_scalar("Total loss per epoch", epoch_loss, epoch)

        # Save final model
    if is_main_process():
        torch.save(model.state_dict(), './checkpoint/' + sub_dir + '/final_model.pth')
        torch.save(model_ema.state_dict(), './checkpoint/' + sub_dir + '/final_model_ema.pth')
        print('Training saved and finished')
        tb.close()
    _barrier()


def training_dif_hints(model, model_ema, dataloader, optimizer, timesteps, channels, image_size):

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=(len(dataloader) * epochs),
    )

    # Writer will output to ./runs/ directory by default
    tb = SummaryWriter('./runs/' + sub_dir + '/', flush_secs=1)
    model.train()
    for epoch in range(epochs):
        model.train()

        if epoch == 0:
            sum_idx = 0
            running_loss = 0.0
        else:
            sum_idx += len(dataloader)
            running_loss = 0.0

        for idx, batch in enumerate(dataloader):
            step_start_time = time.time()
            optimizer.zero_grad()

            batch_size = batch[0].shape[0]
            original = batch[0].to(dtype=torch.float).cuda()
            sketch = batch[1].to(dtype=torch.float).cuda()
            hints = batch[2].to(dtype=torch.float).cuda()
            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=None).long().cuda()

            loss = p_losses_hints(model, original, sketch, hints, t, loss_type="huber")

            if idx % 50 == 0:
                print(
                    "Epoch[{}/{}]({}/{}): Loss_train: {:.4f}, lr_step: {:.4f}, Step time: {} seconds".format(
                        epoch, epochs, idx, len(dataloader),
                        loss.item(), lr_scheduler.get_last_lr()[0], time.time() - step_start_time))

            running_loss = running_loss + loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            ema_update_one(model, model_ema, 0.95 if epoch < 20 else ema_decay)

            # Write some images in tensorboard every save_train_img step
            if (epoch % 5 == 0) and (idx % save_model_train == 0):
                # torch.save(model.state_dict(), './checkpoint/' + sub_dir +'/checkpoint_'+str(idx + sum_idx)+'.pth')
                torch.save(model_ema.state_dict(), './checkpoint/' + sub_dir +'/checkpoint_ema_'+str(idx + sum_idx)+'.pth')

                # save model checkpoint
                torch.save({
                        'epoch': epochs,
                        'model_state_dict': model.state_dict(),
                        'model_state_dict_ema': model_ema.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                }, './checkpoint/' + sub_dir +'/checkpoint_resumen.pth')
                #
            if (idx % save_img==0):
                    # samples = inference_hints_DDIM_in_training(model_ema, feat, batch_size)
                    samples_orig = torchvision.utils.make_grid(reverse_transform_torch(original))
                    samples_sketch = torchvision.utils.make_grid(reverse_transform_torch(sketch))
                    samples_hint = torchvision.utils.make_grid((reverse_transform_torch(hints[:,0:3,:, :])))
                    # samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))
                    tb.add_image('Color in step', samples_orig, idx + sum_idx)
                    tb.add_image('Sketch in step', samples_sketch, idx + sum_idx)
                    tb.add_image('Hint mask in step', samples_hint, idx + sum_idx)
                    # tb.add_image('Sample in step', samples_grid, idx + sum_idx)

            # Writing losses in tensorboard
            tb.add_scalar("Total loss per step", running_loss / (idx + 1), idx + sum_idx)


            tb.flush()

        # Average loss for each epoch
        epoch_loss = running_loss / len(dataloader)

        tb.add_scalar("Total loss per epoch", epoch_loss, epoch)

        # Save final model

    torch.save(model.state_dict(), full_checkpoint_dir)
    torch.save(model_ema.state_dict(), full_checkpoint_dir_ema)
    print('Training saved and finished')
    tb.close()

