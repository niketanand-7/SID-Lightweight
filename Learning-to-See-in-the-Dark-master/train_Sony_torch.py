# train_Sony.py  (refactored to PyTorch)

import os, glob, rawpy, numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from TinyU-Net import TinyUNet   # make sure this file is on PYTHONPATH

# ---------- 1) pack_raw helper (exact same logic) ----------
def pack_raw(raw):
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)
    H, W = im.shape
    im = im.reshape(H, W, 1)
    out = np.concatenate([
        im[0:H:2, 0:W:2], im[0:H:2, 1:W:2],
        im[1:H:2, 1:W:2], im[1:H:2, 0:W:2]
    ], axis=2)
    return out  # H/2 × W/2 × 4

# ---------- 2) Inline Dataset ----------
class SonyRawDataset(Dataset):
    def __init__(self, root='./dataset/Sony', train=True, patch=512):
        self.short_dir = os.path.join(root, 'short')
        self.long_dir  = os.path.join(root, 'long')
        split = '0' if train else '1'
        self.ids = [int(os.path.basename(p)[:5])
                    for p in glob.glob(f'{self.long_dir}/{split}*.ARW')]
        self.patch = patch

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        in_path = glob.glob(f'{self.short_dir}/{_id:05d}_00*.ARW')[0]
        gt_path = glob.glob(f'{self.long_dir}/{_id:05d}_00*.ARW')[0]

        in_exp = float(os.path.basename(in_path)[9:-5])
        gt_exp = float(os.path.basename(gt_path)[9:-5])
        ratio  = min(gt_exp / in_exp, 300)

        raw_in = rawpy.imread(in_path)
        raw_gt = rawpy.imread(gt_path)

        inp = pack_raw(raw_in) * ratio
        gt_img = raw_gt.postprocess(use_camera_wb=True,
                                    no_auto_bright=True,
                                    output_bps=16).astype(np.float32) / 65535.0

        # random crop & flips
        H, W = inp.shape[:2]
        y, x = np.random.randint(0, H-self.patch), np.random.randint(0, W-self.patch)
        inp, gt_img = inp[y:y+self.patch, x:x+self.patch], gt_img[2*y:2*y+2*self.patch, 2*x:2*x+2*self.patch]

        if np.random.rand() < 0.5: inp, gt_img = inp[:, ::-1], gt_img[:, ::-1]
        if np.random.rand() < 0.5: inp, gt_img = inp[::-1, :], gt_img[::-1, :]
        if np.random.rand() < 0.5: inp, gt_img = inp.transpose(1,0,2), gt_img.transpose(1,0,2)

        # to tensor, channels-first
        inp = torch.from_numpy(inp).permute(2,0,1).clamp(0,1)
        gt  = torch.from_numpy(gt_img).permute(2,0,1)
        return inp.float(), gt.float()

# ---------- 3) Set up dataloader, model, optimizer, loss ----------
train_ds = SonyRawDataset(train=True)
loader   = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model   = TinyUNet(in_channels=4, num_classes=3).to(device)
opt     = Adam(model.parameters(), lr=1e-4)
criterion = nn.L1Loss()

# ---------- 4) Training loop ----------
save_freq = 500
for epoch in range(1, 4001):
    if epoch == 2001:
        for g in opt.param_groups: g['lr'] = 1e-5

    for step, (inp, gt) in enumerate(loader, 1):
        inp, gt = inp.to(device), gt.to(device)
        out     = model(inp)
        loss    = criterion(out, gt)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 20 == 0:
            print(f'E{epoch} | Step {step} | Loss {loss.item():.4f}')

    if epoch % save_freq == 0:
        torch.save(model.state_dict(), f'./result_Sony/model_{epoch:04d}.pth')
