# PROPOSED_ALIGN_V1 applied (20260622_164204)
# INFER_64x64_PATCH_V1 applied (20260621_180131)
#!/usr/bin/env python3
# infer_det_proposed.py — DetFill (R3-2 deterministic) inference with PROPOSED-style
# deterministic, region-sorted hints (the piece missing from test_wacv/CreateTestLoader).
#
# Hint generation is a faithful port of:
#   * data_load_det.MyData_train_scrib_det.__getitem__ (the proven region-sort block)
#   * BBDM datasets/custom_64.py:380-389 proposed-eval branch
#       cand = cand[argsort(-count)]; length_samples = int(len(cand) * sample_ratio)
# with the two TRAIN-time randomizations made deterministic for eval:
#   rand_val -> ratio (alpha) ;  random sketch rnd -> sketch_index ;  random dilation -> DIL
#
# Output layout is B_eval_dense_curve-ready:
#   <out_dir>/<sketch_index>/<ratio>/200/<id>.image.png           (generated)
#   <out_dir>/<sketch_index>/<ratio>/ground_truth/<id>.image.png  (GT color)
#
# Run in the `diffusart` conda env (kornia + diffusers + torch).
import os, argparse

p = argparse.ArgumentParser()
p.add_argument('--checkpoint_path', required=True, help='det EMA checkpoint .pth')
p.add_argument('--hint', default='strokes', choices=['strokes', 'scribble', 'dot'])
p.add_argument('--ratio', type=float, required=True, help='hint ratio alpha (0..1)')
p.add_argument('--sketch_index', type=int, default=0, help='0=pysimp 1=XDoG 2=sketchkeras')
p.add_argument('--domain', default='illust')
p.add_argument('--dtype', default='felzenszwalb')
p.add_argument('--data_root', default='/scratch/madorin/diffusart_det_R3-2', help='hint/sketch/GT root (host-local /scratch; has all 3 sketches incl pysimp)')
p.add_argument('--region_root', default='/scratch/madorin/diffusart_det_R3-2/illust/hint_from_regions/felzenszwalb',
               help='flat dir of <fname>.image_region64.png (proposed-eval region source)')
p.add_argument('--region_suffix', default='.image_region64.png')
p.add_argument('--test_list', required=True, help='configs/<domain>/test_paper.txt')
p.add_argument('--out_dir', required=True)
p.add_argument('--gpu', default='0')
p.add_argument('--num_steps', type=int, default=100)
p.add_argument('--batch_size', type=int, default=16)
p.add_argument('--limit', type=int, default=0, help='debug: only first N images')
p.add_argument('--start_idx', type=int, default=0)
p.add_argument('--dilation', type=int, default=3)
args = p.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
# DATA_ROOT must be set BEFORE importing data_load_det (read at import time)
os.environ['DIFFUSART_DET_DATA_ROOT'] = args.data_root

import cv2, numpy as np, torch
from collections import OrderedDict
from skimage.transform import resize
from skimage import io
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from diffusers import DPMSolverMultistepScheduler

from utils import *                              # reverse_transform_torch, etc.
from models.model_coupled_v1 import Unet
from data.data_load_det import MyData_train_scrib_det, pil2opencv, opencv2pil

def reverse_transform_torch(x):
    # de-normalize [-1,1] -> [0,1] for save_image (matches utils.py:511 (x/2+0.5).clamp(0,1))
    return (x / 2 + 0.5).clamp(0, 1)

DEVICE = torch.device('cuda')
IMG = 256


class DetEvalData(MyData_train_scrib_det):
    """Deterministic, region-sorted, proposed-style hint loader for eval."""
    def __init__(self, root, hint, ratio, sketch_index, domain, dtype, dilation,
                 region_root, region_suffix):
        super().__init__(root, IMG, approach='proposed', hint=hint,
                         domain=domain, dtype=dtype, dtype_region=dtype)
        self.ratio = float(ratio)
        self.sketch_index = int(sketch_index)
        self.dil = int(dilation)
        self.region_root = region_root
        self.region_suffix = region_suffix
        # PROPOSED_ALIGN_V1: GT/sketch transforms matching training (BICUBIC+Normalize)
        self._gt_transform = transforms.Compose([
            transforms.Resize((IMG, IMG), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self._sk_transform = transforms.Compose([
            transforms.Resize((IMG, IMG), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __getitem__(self, index):
        sk = self.sketch_index
        path = self.root[index].replace('\n', '')
        fname = path.split('/')[-1].split('.png')[0].split('.jpg')[0].split('.image')[0]
        dname = path.split('/')[-2]

        original = self.load_image(os.path.join(self.root_region, dname, fname + '.image.png'))  # PROPOSED_ALIGN_V1 PIL
        try:
            sketch_data = self.sketch_loader(os.path.join(self.root_sketch, self.sketch_cands[sk],
                                                          dname, fname + '.png'))  # PIL L
        except Exception:
            sketch_data = Image.fromarray(np.zeros((IMG, IMG), dtype=np.uint8))

        # region map: proposed-eval source = flat <region_root>/<fname>.image_region64.png
        rpath = os.path.join(self.region_root, fname + self.region_suffix)
        if not os.path.isfile(rpath):
            rpath = os.path.join(self.region_root, dname, fname + self.region_suffix)
        region = cv2.imread(rpath)
        if region is None:
            raise FileNotFoundError(f'region map not found: {rpath}')
        region = cv2.resize(region, (256, 256), interpolation=cv2.INTER_NEAREST)
        r32 = region.astype(np.int32)
        _packed = (r32[:, :, 0] << 16) | (r32[:, :, 1] << 8) | r32[:, :, 2]
        cand_p, count = np.unique(_packed.ravel(), return_counts=True)
        cand_p = cand_p[np.argsort(-count)]                       # region sort (size desc)
        length_samples = int(len(cand_p) * self.ratio)            # DETERMINISTIC (alpha)

        single_hint = np.zeros((region.shape[0], region.shape[1], 3))
        single_mask = np.zeros((region.shape[0], region.shape[1], 1))
        _top = cand_p[:length_samples]
        _lut = np.zeros(256, dtype=bool)
        if len(_top):
            _lut[np.concatenate([_top >> 16, (_top >> 8) & 255, _top & 255])] = True
        _sel = _lut[region[:, :, 0]] & _lut[region[:, :, 1]] & _lut[region[:, :, 2]]
        single_hint[_sel] = 1
        single_mask[_sel[:, :, None]] = 1

        if self.hint == 'dot':
            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_dot_mask256.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_dot_col256.png'))
        else:  # strokes / scribble
            mask = self.sketch_loader(os.path.join(self.root_scr, dname, fname + '.image_scribble_mask256.png'))
            hint = self.load_image(os.path.join(self.root_scr, dname, fname + '.image_scribble_col256.png'))

        hint = pil2opencv(hint)
        mask = pil2opencv(mask)
        # INFER_64x64: NN-upsample hint/mask 64->256 to match training-time data_load_det_64.py
        hint = cv2.resize(hint, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        single_hint = single_hint * hint
        single_mask = single_mask * mask[:, :, np.newaxis]
        # DILATE_REMOVED_V1: dilate+blur removed to match dilate-free retraining (20260622_160051)
        # kernel = np.ones((self.dil, self.dil), np.uint8)
        # single_hint = cv2.dilate(single_hint, kernel=kernel, iterations=1)
        # single_mask = cv2.dilate(single_mask, kernel=kernel, iterations=1)
        # single_mask = cv2.GaussianBlur(single_mask, ksize=(3, 3), sigmaX=0)
        hint = opencv2pil(single_hint.astype(np.uint8))
        mask = opencv2pil(single_mask[:, :, 0].astype(np.uint8))  # MASK_SQUEEZE_FIX
        totensor = transforms.ToTensor()
        mask = totensor(mask).float()
        hint = totensor(hint).float()
        hint = (hint - 0.5) / 0.5
        hint = torch.cat([mask, hint * mask], dim=0)  # PROPOSED_ALIGN_V1 mask-first

        # region map (segmentation) used, as [0,1] RGB tensor for inspection-saving
        region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        region_t = torch.from_numpy(region_rgb.copy()).permute(2, 0, 1).float() / 255.0
        # PROPOSED_ALIGN_V1: PIL Resize(NEAREST->IMG) then BICUBIC+Normalize (matches training, no flip at eval)
        original = transforms.Resize((IMG, IMG), Image.NEAREST)(original)
        sketch_data = transforms.Resize((IMG, IMG), Image.NEAREST)(sketch_data)
        original = self._gt_transform(original)
        sketch_data = self._sk_transform(sketch_data)
        return original, sketch_data, hint, fname, region_t


@torch.no_grad()
def sample_dpm(model, sketch, hints, scheduler, num_steps):
    scheduler.set_timesteps(num_inference_steps=num_steps)   # reset scheduler state per image
    b = sketch.shape[0]
    feat = torch.cat((sketch.to(DEVICE).float(), hints.to(DEVICE).float()), dim=1)   # 5ch
    torch.manual_seed(2)
    noise = torch.randn((b, 8, IMG, IMG), device=DEVICE)
    img = noise[:, 5:, :, :]
    for t in scheduler.timesteps:
        t = t.to(DEVICE).float()
        t_tensor = torch.tensor([t]).to(DEVICE).float()
        x_in = torch.cat((feat, img), dim=1)
        residual = model(x_in, feat, t_tensor)
        img = scheduler.step(residual, t.long(), img).prev_sample
    return img


def main():
    model = Unet(dim=IMG, channels=8, dim_mults=(1, 2)).to(DEVICE)
    sd = torch.load(args.checkpoint_path, map_location=DEVICE)
    # INFER_64x64: unwrap common wrapper-dict checkpoint formats
    if isinstance(sd, dict) and not any(isinstance(v, torch.Tensor) for v in sd.values()):
        for _key in ('model_state_dict_ema', 'ema', 'model_ema', 'state_dict', 'model_state_dict', 'model', 'net'):
            if _key in sd and isinstance(sd[_key], dict):
                print(f'[infer] unwrapped checkpoint key: {_key}', flush=True)
                sd = sd[_key]
                break
    nsd = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in sd.items())
    model.load_state_dict(nsd)
    model.eval()
    print(f'[infer] loaded {args.checkpoint_path}')

    with open(args.test_list) as f:
        raw_items = [l for l in f.readlines() if l.strip()]
    # INFER_64x64: rewrite legacy /home/madorin/datasets/.../main_exp_felzenszwalb/ paths to /scratch
    _OLD_PFX = '/home/madorin/datasets/tog2024/main_exp_felzenszwalb_tmp/'
    _OLD_PFX2 = '/home/madorin/datasets/tog2024/main_exp_felzenszwalb/'
    _NEW_PFX = '/scratch/madorin/diffusart_det_R3-2/'
    items = []
    for _l in raw_items:
        _l = _l.replace(_OLD_PFX, _NEW_PFX).replace(_OLD_PFX2, _NEW_PFX)
        items.append(_l)
    items = items[args.start_idx:]
    ds = DetEvalData(items, args.hint, args.ratio, args.sketch_index, args.domain, args.dtype, args.dilation,
                     args.region_root, args.region_suffix)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=2, shuffle=False)

    rs = str(args.ratio)
    base = os.path.join(args.out_dir, str(args.sketch_index), rs)
    gen_dir = os.path.join(base, '200')              # B_eval reads these two
    gt_dir = os.path.join(base, 'ground_truth')
    hint_dir = os.path.join(base, 'hint')            # inspection: colored hint input
    region_dir = os.path.join(base, 'region')        # inspection: segmentation/region map used
    sketch_dir = os.path.join(base, 'sketch')        # inspection: sketch input
    for d in (gen_dir, gt_dir, hint_dir, region_dir, sketch_dir):
        os.makedirs(d, exist_ok=True)

    sched = DPMSolverMultistepScheduler(beta_schedule='linear', beta_start=1e-4,
                                        algorithm_type='dpmsolver++', solver_order=2,
                                        num_train_timesteps=1000, thresholding=True)
    sched.set_timesteps(num_inference_steps=args.num_steps)

    done = 0
    for color, sketch, hints, fname, region_t in loader:
        if args.limit and done >= args.limit:
            break
        names = list(fname)
        # resume: skip the whole batch only if every item is already generated
        if all(os.path.isfile(os.path.join(gen_dir, f'{n}.image.png')) for n in names):
            done += len(names)
            continue
        out = reverse_transform_torch(sample_dpm(model, sketch, hints, sched, args.num_steps))
        col = reverse_transform_torch(color.to(DEVICE).float())
        hin = reverse_transform_torch(hints[:, :3].to(DEVICE).float())
        skt = reverse_transform_torch(sketch.to(DEVICE).float())
        for j, n in enumerate(names):
            save_image(out[j].unsqueeze(0), os.path.join(gen_dir, f'{n}.image.png'))
            save_image(col[j].unsqueeze(0), os.path.join(gt_dir, f'{n}.image.png'))
            save_image(hin[j].unsqueeze(0), os.path.join(hint_dir, f'{n}.png'))
            save_image(region_t[j].unsqueeze(0), os.path.join(region_dir, f'{n}.png'))
            save_image(skt[j].unsqueeze(0), os.path.join(sketch_dir, f'{n}.png'))
        done += len(names)
        if done % (args.batch_size * 10) < args.batch_size:
            print(f'[infer] sk{args.sketch_index} a{rs}: ~{done} done', flush=True)
    print(f'[infer] DONE sk{args.sketch_index} a{rs}: {done} -> {gen_dir}', flush=True)


if __name__ == '__main__':
    main()
