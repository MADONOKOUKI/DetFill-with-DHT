import torch
from utils import *
from training.forward import *
from torch.profiler import profile, record_function, ProfilerActivity

def p_losses_wo_line(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)


    predicted_noise = denoise_model(x_noisy, t)



    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def p_losses(denoise_model, x_start, sketch, t, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, sketch, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def p_losses_hints(denoise_model, x_start, sketch, hints, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    sketch_hints = torch.cat((sketch, hints), dim=1)
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)

    x_noisy_hints = torch.cat((sketch_hints, x_noisy), dim=1)
    # print(x_noisy_hints.shape, sketch_hints.shape)
    predicted_noise = denoise_model(x_noisy_hints, sketch_hints, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

def p_losses_lines(denoise_model, x_start, sketch, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)


    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    x_cat_noisy = torch.cat((sketch,  x_noisy), dim=1)



    predicted_noise = denoise_model(x_cat_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss