import torch
from utils import *
import skimage.draw
from collections import OrderedDict
# from models.model_atten_unet import Unet
# from models.cond_encoder import cond_encod
from testing.testing import *
# from models.model_coupled import Unet
# from models.model_attent_line_art_v1 import Unet
from models.model_coupled_v1 import Unet
from torchvision import transforms
from data.data_load import *
from data.loader_list import *
from training.training_src import *
from torchviz import make_dot
from prettytable import PrettyTable
from torchinfo import summary

from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage import io, color
from skimage.transform import resize
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


device = "cuda:1" if torch.cuda.is_available() else "cpu"

original_data_val, sketchkeras_val, sketchsimp_val = init_load_data_val()
loader_val = MyData_eval_load(original_data_val, sketchkeras_val, sketchsimp_val, size=256)
# loader_val = MyData_eval_load(original_data_val[3070:3370], sketchkeras_val[3070:3370], sketchsimp_val[3070:3370], size=256)
dataloader_test = DataLoader(loader_val, batch_size=4, num_workers=4, shuffle=False)

fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='sum').to(device)
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction='sum').to(device)
ssim_tot = 0
lpips_tot = 0
l = len(dataloader_test)
with torch.no_grad():
    for idx, batch in enumerate(dataloader_test):
        # print(l)
        #     # fake_cim = batch[1].to(device).to(dtype=torch.float)
        # name = batch[2]
        # real_cim = batch[0].to(device).to(dtype=torch.float)
        # samples_PNDM_grid = torchvision.utils.make_grid(real_cim)
        #
        # save_image(samples_PNDM_grid,
        #        '/home/hec/Phd/Diffusion_v1/test_img/' + name[0])

        name = batch[2]
        fake_cim = batch[1].to(device).to(dtype=torch.float)
        real_cim = batch[0].to(device).to(dtype=torch.float)
        fid.update(real_cim, real=True)
        fid.update(fake_cim, real=False)

        ssim_value = ssim(fake_cim, real_cim)
        ssim_tot += ssim_value

        lpips_value = lpips(fake_cim, real_cim)
        lpips_tot += lpips_value

        print(idx, '/', l, 'ssim:', ssim_value, 'lpips:', lpips_value)

    fid_value = fid.compute()
    ssim_avg = ssim_tot / (l * 4)
    lpips_avg = lpips_tot / (l * 4)

    print('FID_value:', fid_value)
    print('SSIM_value:', ssim_avg)
    print('Lpips_value:', lpips_avg)
    fid.reset()