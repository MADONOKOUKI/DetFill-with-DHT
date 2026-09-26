import matplotlib.pyplot as plt
import torch
import scipy.stats as stats
from collections import OrderedDict
from torch.optim import Adam
from torchvision import transforms
from torchviz import make_dot
from prettytable import PrettyTable
from torchinfo import summary
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from data.data_load import *
from data.loader_list import *
from torchvision.utils import make_grid

from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

# load dataset from the hub
batch_size = 4

transform = Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

reverse_transform_f = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t[0].permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t.to('cpu').numpy()),
])

reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
])


original_data, sketchkeras, sketchsimp = init_load_data()


loader_train = MyData_train(original_data, sketchkeras, sketchsimp)


# create dataloader
dataloader_train = DataLoader(loader_train, batch_size=batch_size, num_workers=5, shuffle=False)

batch = next(iter(dataloader_train))

batch_size = batch[0].shape[0]
original = batch[0]
sketch = batch[1]
hints = batch[2]
print(sketch.shape)
print(hints.shape)


# image = reverse_transform(batch[0])
#
# image_size = 128
# mu, sigma = 1, 0.005
# X = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
#
# def mask_gen():
#     maskS = image_size
#     mask = torch.rand(1, 1, maskS, maskS).ge(X.rvs(1)[0]).float()
#     return mask
#
#
# mask = mask_gen()
# print(mask.shape)
# hint = torch.cat((image * mask, mask), 1)
# print(hint[0].shape)
# grid_hint = make_grid(hint)
# torchvision.utils.save_image(grid_hint, './test3_4.png')

