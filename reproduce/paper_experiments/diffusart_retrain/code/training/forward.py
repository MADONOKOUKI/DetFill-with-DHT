import torch
from utils import *
from tqdm.auto import tqdm
from models.schedulers import *


devices = "cuda:1" if torch.cuda.is_available() else "cpu"

################################################# Forward process #######################################################
timesteps_inf = 1000
timesteps = 1000
# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)
# betas = betas.to(devices)
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

def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

@torch.no_grad()
def p_sample(model, x, feat, t, t_index):
    betas_t = extract_inf(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract_inf(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract_inf(sqrt_recip_alphas, t, x.shape)

    noise_pred = model(x, feat, t)

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean.to(device) + torch.sqrt(posterior_variance_t).to(device) * noise.to(device)

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model, feat, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps_inf)), desc='sampling loop time step', total=timesteps_inf):
        img = p_sample(model, img, feat, torch.full((b,), i, device=device, dtype=torch.long), i)
        # imgs.append(img.cpu().numpy())
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample(model, feat, seed,image_size, batch_size=16, channels=3):
    torch.manual_seed(seed)
    return p_sample_loop(model, feat, shape=(batch_size, channels, image_size, image_size))

######## w/o line art ##############

@torch.no_grad()
def p_sample_wo(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract_inf(sqrt_recip_alphas, t, x.shape)

    noise_pred = model(x, t)
    print(t.max(), t.min())
    print('noise max', x.max(), 'noise min', x.min(), 'mean', x.mean(), 'var', x.std())
    print('pred max', noise_pred.max(), 'pred min', noise_pred.min(), 'mean', x.mean(), 'var', x.std())
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop_wo(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps_inf)), desc='sampling loop time step', total=timesteps_inf):
        img = p_sample_wo(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        # imgs.append(img.cpu().numpy())
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample_wo(model, image_size, batch_size=16, channels=3):
    # torch.manual_seed(seed)
    return p_sample_loop_wo(model, shape=(batch_size, channels, image_size, image_size))


################################### with hints ###########################################################"

@torch.no_grad()
def p_sample_hints(model, x_in, feat, t, t_index, cat):
    noise_pred = model(x_in, feat, t.to(devices))
    if cat == True:
        x = x_in[:, 1::, :, :]
    else:
        x = x_in
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
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.to(devices)
    betas_t = betas_t.to(devices)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(devices)
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
        noise = noise.to(devices)
        sqrt_var = torch.sqrt(posterior_variance_t)
        sqrt_var = sqrt_var.to(devices)
        # Algorithm 2 line 4:
        return model_mean + sqrt_var * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop_hints(model, noise, feat, hints, shape, cat=None):
    # device = next(model.parameters()).device
    b = shape[0]
    sketch = feat
    feat = torch.cat((feat[0:b], hints[0:b]), dim=1)
    # start from pure noise (for each example in the batch)
    if cat == True:
        img = noise[:, 1::, :, :]
    else:
        img = noise
    imgs = []

    for i in tqdm(reversed(range(0, timesteps_inf)), desc='sampling loop time step', total=timesteps_inf):
        if cat == True:
            img = torch.cat((sketch[0:b], img[0:b]), dim=1)

        img = p_sample_hints(model, img, feat, torch.full((b,), i, dtype=torch.long), i, cat)
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample_hints(model, noise, feat, hints,image_size, batch_size=16, channels=3, cat=None):
    return p_sample_loop_hints(model, noise, feat, hints, shape=(batch_size, channels, image_size, image_size), cat=cat)


################################################ VAE #############################################################
@torch.no_grad()
def p_sample_hints_vae(model, x_in,  t, t_index):
    noise_pred = model(x_in, t.cuda())
    x = x_in[:, 4::, :, :]
    # print('in', x_in.shape)
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    # print(x.shape, feat.shape)

    # print(t.max(), t.min())
    # print('pred max', noise_pred.max(), 'pred min', noise_pred.min(), 'mean', x.mean(), 'var', x.std())
    # print('proper_pred max', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).max(), 'pred min', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).min(), 'mean', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).mean(), 'var', (betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t).std())
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.cuda()
    betas_t = betas_t.cuda()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.cuda()
    # print(x.shape, noise_pred.shape)
    # noise_pred = torch.cat((feat[:, 0:1, :, :], noise_pred), dim=1)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        noise = noise.cuda()
        sqrt_var = torch.sqrt(posterior_variance_t)
        sqrt_var = sqrt_var.cuda()
        # Algorithm 2 line 4:
        return model_mean + sqrt_var * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop_hints_vae(model, noise, sketch, shape, cat=None):
    # device = next(model.parameters()).device
    b = shape[0]
    # start from pure noise (for each example in the batch)
    img= noise
    imgs = []

    for i in tqdm(reversed(range(0, timesteps_inf)), desc='sampling loop time step', total=timesteps_inf):
        if cat == True:
            img = torch.cat((sketch[0:b], img[0:b]), dim=1)


        img = p_sample_hints_vae(model, img, torch.full((b,), i, dtype=torch.long), i)
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample_hints_vae(model, noise, feat, image_size, batch_size=16, channels=3, cat=None):
    return p_sample_loop_hints_vae(model, noise, feat,  shape=(batch_size, channels, image_size, image_size), cat=cat)









@torch.no_grad()
def p_sample_hints_cat(model, x_in, feat, t, t_index, cat):
    noise_pred = model(x_in, feat, t.to(devices))

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
    sqrt_recip_alphas_t = sqrt_recip_alphas_t.to(devices)
    betas_t = betas_t.to(devices)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.to(devices)
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
        noise = noise.to(devices)
        sqrt_var = torch.sqrt(posterior_variance_t)
        sqrt_var = sqrt_var.to(devices)
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

    for i in tqdm(reversed(range(0, timesteps_inf)), desc='sampling loop time step', total=timesteps_inf):

        img = torch.cat((feat[0:b], img[0:b]), dim=1)

        img = p_sample_hints_cat(model, img, feat, torch.full((b,), i, dtype=torch.long), i, cat)
        imgs.append(img.cpu())
    return imgs


@torch.no_grad()
def sample_hints_cat(model, noise, feat, hints,image_size, batch_size=16, channels=3, cat=None):
    return p_sample_loop_hints_cat(model, noise, feat, hints, shape=(batch_size, channels, image_size, image_size), cat=cat)

