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
# === R5 INSTRUMENTATION ===
import torchvision
from torchvision.utils import save_image as _save_image_R5
from diffusers import DPMSolverMultistepScheduler as _DPMS_R5
_R5_PROBE = {"hook_act": None, "cond_grad_norm": None, "val_batch": None,
             "abort_flag_path": os.environ.get("R5_ABORT_FLAG", "/tmp/R5_ABORT")}
def _r5_hook(m, inp, out):
    _R5_PROBE["hook_act"] = out.detach() if isinstance(out, torch.Tensor) else (out[0].detach() if hasattr(out,"__getitem__") else None)
def _r5_load_val_batch(device):
    if _R5_PROBE["val_batch"] is not None: return _R5_PROBE["val_batch"]
    try:
        from data.data_load_det_64 import MyData_train_scrib_det_64
        with open("configs/illust/valid_paper.txt") as f:
            items = [l.strip() for l in f if l.strip()][:4]
        ds = MyData_train_scrib_det_64(items, size=256, approach="proposed", hint="strokes",
                                        domain="illust", dtype="felzenszwalb", dtype_region="felzenszwalb")
        bs = [ds[i] for i in range(min(4, len(items)))]
        gt = torch.stack([b[0] for b in bs]).to(device)
        sk = torch.stack([b[1] for b in bs]).to(device)
        hi = torch.stack([b[2] for b in bs]).to(device)
        _R5_PROBE["val_batch"] = (gt, sk, hi)
        return _R5_PROBE["val_batch"]
    except Exception as e:
        print("[R5] val batch load failed:", e, flush=True); return None
def _r5_sample_dpm(model, sketch, hints, device, num_steps=25):
    feat = torch.cat((sketch, hints), dim=1)
    sched = _DPMS_R5(beta_schedule="linear", beta_start=1e-4, algorithm_type="dpmsolver++",
                    solver_order=2, num_train_timesteps=1000, thresholding=True)
    sched.set_timesteps(num_inference_steps=num_steps)
    b = sketch.shape[0]
    torch.manual_seed(2)
    noise = torch.randn((b, 8, 256, 256), device=device)
    img = noise[:, 5:, :, :].clone()
    with torch.no_grad():
        for t in sched.timesteps:
            t = t.to(device).float(); tt = torch.tensor([t]).to(device).float()
            x_in = torch.cat((feat, img), dim=1)
            residual = model(x_in, feat, tt)
            img = sched.step(residual, t.long(), img).prev_sample
    return img
def _r5_probe(model_ema, epoch, tb, sub_dir, device):
    """returns (delta_cond, pairwise_cosim) and writes images/scalars to TB"""
    vb = _r5_load_val_batch(device)
    if vb is None: return None, None
    gt, sk, hi = vb
    m = model_ema.eval()
    if hasattr(m, "module"): m = m.module
    pred_real = _r5_sample_dpm(m, sk, hi, device)
    pred_zero = _r5_sample_dpm(m, torch.zeros_like(sk), torch.zeros_like(hi), device)
    delta = (pred_real - pred_zero).abs().mean().item()
    pflat = pred_real.flatten(1); pflat = pflat / (pflat.norm(dim=1, keepdim=True) + 1e-8)
    cs = (pflat @ pflat.t()); n = pflat.shape[0]
    if n > 1:
        cosim = (cs.sum() - n) / (n*(n-1))
        cosim = float(cosim)
    else: cosim = 1.0
    print(f"[R5 PROBE] ep{epoch}: delta_cond={delta:.4f}  pairwise_cosim={cosim:.4f}", flush=True)
    tb.add_scalar("probe/delta_cond", delta, epoch)
    tb.add_scalar("probe/pairwise_cosim", cosim, epoch)
    tb.add_image("val/pred_real_ema", torchvision.utils.make_grid(((pred_real/2+0.5).clamp(0,1))), epoch)
    tb.add_image("val/pred_zero_ema", torchvision.utils.make_grid(((pred_zero/2+0.5).clamp(0,1))), epoch)
    tb.add_image("val/gt", torchvision.utils.make_grid(((gt/2+0.5).clamp(0,1))), epoch)
    return delta, cosim
def _r5_check_abort(epoch, delta, cosim):
    """HARD ABORT GATES (per workflow synth)"""
    if delta is None: return False
    abort = False; reason = ""
    if epoch == 10 and (delta < 0.05 or cosim > 0.90): abort=True; reason=f"ep10 gate: delta={delta:.3f}<0.05 or cosim={cosim:.3f}>0.90"
    elif epoch == 25 and (delta < 0.10 or cosim > 0.85): abort=True; reason=f"ep25 gate: delta={delta:.3f}<0.10 or cosim={cosim:.3f}>0.85"
    elif epoch == 50 and delta < 0.15: abort=True; reason=f"ep50 gate: delta={delta:.3f}<0.15"
    if abort:
        print(f"[R5 ABORT] {reason}  — writing flag and exiting", flush=True)
        open(_R5_PROBE["abort_flag_path"],"w").write("epoch="+str(epoch)+" "+reason+chr(10))
        return True
    return False
# === END R5 ===

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

    # === R5: register cond_encod forward hook ===
    _root = model.module if hasattr(model, "module") else model
    try:
        _root.encod_sketch.register_forward_hook(_r5_hook)
        print("[R5] forward hook on encod_sketch registered", flush=True)
    except Exception as _e:
        print("[R5] hook register failed:", _e, flush=True)
    # === END R5 hook ===
    model.train()
    for epoch in range(epochs):
        # === R5: per-epoch probe + abort gate (rank0) ===
        if is_main_process() and (epoch % 5 == 0 or epoch in (10, 25, 50)):
            try:
                delta, cosim = _r5_probe(model_ema, epoch, tb, sub_dir, torch.device("cuda"))
                if _r5_check_abort(epoch, delta, cosim):
                    _barrier()
                    import sys; sys.exit(2)
            except Exception as _e:
                print(f"[R5] probe failed @ep{epoch}: {_e}", flush=True)
        if os.path.exists(_R5_PROBE["abort_flag_path"]):
            print(f"[R5] abort flag detected, exiting rank", flush=True)
            _barrier(); import sys; sys.exit(2)
        # === END R5 probe ===
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
            # === R5: cond_encod activation log every 200 steps ===
            if is_main_process() and idx % 200 == 0:
                _a = _R5_PROBE.get("hook_act")
                if _a is not None and isinstance(_a, torch.Tensor):
                    tb.add_scalar("diag/cond_act_mean_abs", float(_a.abs().mean()), idx + sum_idx)
                    tb.add_scalar("diag/cond_act_std", float(_a.std()), idx + sum_idx)
                    tb.add_scalar("diag/cond_act_frac_zero", float((_a.abs() < 1e-6).float().mean()), idx + sum_idx)
            # === END R5 act_log ===


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

