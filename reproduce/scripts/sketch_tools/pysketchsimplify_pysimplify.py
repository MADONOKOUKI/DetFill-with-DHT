import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
from PIL import Image
import skimage.transform
import glob, os
import torch
from torchvision import transforms

from PIL import Image

class Simplifier:
   def __init__(self, model='aki_edit_d', weights='models/aki_edit_d_l1_w2_iter1000000.pth.tar', use_gpu=True, immean=0.96644231074546):
      SketchNet = __import__( model, fromlist=['SketchNet'] ).SketchNet
      self.use_gpu = use_gpu
      self.immean = immean
      self.model = SketchNet()
      map_location = None if self.use_gpu else lambda storage, loc: storage
      self.model.load_state_dict( torch.load(weights, map_location=map_location)['model'] )
      if self.use_gpu:
         self.model.cuda()
      self.model.bnabsorb()
      #print(self.model)
      self.model.eval()

   def __call__(self, data_in, edit_in=None ):
      data_in = Image.open(data_in).convert('L')
      ow, oh  = data_in.size[0], data_in.size[1]
      data_in = transforms.Resize((1280, 1280))(data_in)
      #data_in = transforms.Resize((1152, 1152))(data_in)
      #data_in = transforms.Resize((1024, 1024))(data_in)
      #data_in = transforms.Resize((768, 768))(data_in)
      w, h = data_in.size[0], data_in.size[1]
      pw    = 8-(w%8) if w%8!=0 else 0
      ph    = 8-(h%8) if h%8!=0 else 0
      data  = (transforms.ToTensor()(data_in)-self.immean).unsqueeze(0)
      if edit_in is None:
         edit = torch.zeros( data.size() )
      elif isinstance(edit_in, type(Image)): # Image
         edit = edit_in.resize_( data_in.size )
         edit = transforms.ToTensor()(edit).unsqueeze(0)
         edit = edit*(-2.)+1.
      else: # numpy
         edit = skimage.transform.resize( edit_in, (data.size(2),data.size(3)), mode='reflect' )
         edit = transforms.ToTensor()(edit).unsqueeze(0)
         save_image( edit/2+0.5, 'user_raw.png' )
      data  = torch.cat( [data.float(),edit.float()], 1 )
      if self.use_gpu:
         data = data.cuda()
      with torch.no_grad():
         if pw!=0 or ph!=0:
            data = torch.nn.functional.pad( data, pad=(0,pw,0,ph), mode='replicate' )
         pred=self.model.forward( data )[0,:,0:h,0:w].data.float()
         pred =pred.cpu()
         transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((oh, ow)),
            transforms.ToTensor()
         ])
         pred = [transform(pred_) for pred_ in pred]
         filepath, filename = os.path.split(files1)
         save_image(pred[0], os.path.join(out_dir, filename))


simp=Simplifier()
# in_dir = 'in'
# out_dir = 'out4'
in_dir = '/home/USER/gitlab/yuan/lac_eval_v1-main/DanbooRegion2020_sketch/sketchkeras/val'
# out_dir = 'sketchkeras_enhanced'
out_dir = '/home/USER/gitlab/yuan/lac_eval_v1-main/DanbooRegion2020_sketch/pysketchsimplify/val'    

if not os.path.exists(out_dir): os.mkdir(out_dir)
for files1 in glob.glob(in_dir + '/*.jpg'):
   simp(files1)