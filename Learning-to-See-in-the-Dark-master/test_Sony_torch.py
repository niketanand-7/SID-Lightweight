# test_Sony.py  (PyTorch version)
import os, glob, rawpy, numpy as np, torch, matplotlib.pyplot as plt
from TinyU-Net import TinyUNet, pack_raw

model = TinyUNet(in_channels=4, num_classes=3)
model.load_state_dict(torch.load('./result_Sony/model_4000.pth'))
model.eval().cuda()

input_dir, gt_dir = './dataset/Sony/short/', './dataset/Sony/long/'
out_dir = './result_Sony/final/'; os.makedirs(out_dir, exist_ok=True)
test_ids = [int(os.path.basename(p)[:5]) for p in glob.glob(gt_dir + '/1*.ARW')]

for tid in test_ids:
    in_path = glob.glob(f'{input_dir}/{tid:05d}_00*.ARW')[0]
    gt_path = glob.glob(f'{gt_dir}/{tid:05d}_00*.ARW')[0]
    in_exp, gt_exp = map(lambda fn: float(os.path.basename(fn)[9:-5]), (in_path, gt_path))
    ratio = min(gt_exp/in_exp, 300)

    raw_in = rawpy.imread(in_path)
    inp = pack_raw(raw_in)[None] * ratio
    inp = torch.from_numpy(inp).permute(0,3,1,2).to('cuda')

    with torch.no_grad():
        out = model(inp).clamp(0,1)[0].permute(1,2,0).cpu().numpy()

    raw_gt = rawpy.imread(gt_path)
    gt_img = raw_gt.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=16)/65535.0

    fn = f'{tid:05d}_00_{int(ratio)}'
    plt.imsave(f'{out_dir}/{fn}_out.png', out)
    plt.imsave(f'{out_dir}/{fn}_gt.png', gt_img)
