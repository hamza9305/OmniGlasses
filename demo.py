from Exr_Files_Pytorch import import_exr
from PIL import Image

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

TO_TENSOR = transforms.ToTensor()
TO_PIL = transforms.ToPILImage()

def save_tensor(tensor, path):
    pil_img = TO_PIL(tensor[0])
    pil_img.save(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Demo: View synthesis with LUTs')
    #parser.add_argument('--path', default='masks_and_luts', required=True)
    parser.add_argument('--path', default='masks_and_luts')
    parser.add_argument('--max_disp_deg', type=float, default=None, help="maximum allowed disparity in degrees (cannot be used with --max_disp_rad)")
    parser.add_argument('--max_disp_rad', type=float, default=None, help="maximum allowed disparity in radians [default: 0.31]")
    parser.add_argument('--max_disp_at_index', type=int, default=176, help="index of lut slice corresponding to maximum disparity")
    parser.add_argument('--plot', action='store_true', help="Plot debug plots")
    parser.add_argument('--save_plots', action='store_true', help="Save debug plots as svg files [needs --plot]")
    parser.add_argument('--left_image', '-l', default='demo_res/0000977_img.webp', help="left camera's image")
    parser.add_argument('--right_image', '-r', default='demo_res/0000977_img_stereo.webp', help="right camera's image")
    parser.add_argument('--disparity', '-d', default='demo_res/0000977_disp.exr', help="right camera's image")
    parser.add_argument('--device', default='cuda:0', help="tensor processing device, e.g. cuda:0")

    args = parser.parse_args()

    max_disp_rad = 0.31 # default
    if args.max_disp_deg is not None and args.max_disp_rad is not None:
        print('Enter only one max_disp_deg or max_disp_rad or none of them.')
        exit(1)
    elif args.max_disp_rad is not None:
        max_disp_rad = args.max_disp_rad
    elif args.max_disp_deg is not None:
        max_disp_rad = args.max_disp_deg * math.pi / 180

    #paths
    path_stages = os.path.join(args.path, "stage%d")
    path_mask = path_stages + '/' + "mask_stage%d.pt"
    path_lut_stack = path_stages + '/' + "lut_stage%d.pt"

    stage = 3
    
    cur_path_lut_stack = path_lut_stack % (stage, stage)
    cur_path_mask = path_mask % (stage, stage)
    lut_stack = torch.load(cur_path_lut_stack, map_location=args.device)
    lut_stack = torch.squeeze(lut_stack)
    inv_mask = torch.load(cur_path_mask, map_location=args.device)

    img_left = Image.open(args.left_image)
    img_left = TO_TENSOR(img_left)[None] # [1, 3, H, W]
    img_right = Image.open(args.right_image)
    img_right = TO_TENSOR(img_right)[None]
    disp = import_exr(args.disparity)
    disp = torch.from_numpy(disp)

    step_size = max_disp_rad / args.max_disp_at_index
    disp_index = disp / step_size
    disp_index = torch.round(disp_index).type(torch.long)
    disp_index = disp_index[None, None]

    inv_mask_rgb = inv_mask[None, None, :, :, 0].repeat([1, 3, 1, 1])

    img_left[inv_mask_rgb] = float('nan')
    img_right[inv_mask_rgb] = float('nan')
    disp_index[inv_mask_rgb[:, 0:1]] = 0
    disp_index[disp_index < 0] = 0
    disp_index[disp_index > args.max_disp_at_index] = 0

    no_steps = lut_stack.shape[0]
    height = lut_stack.shape[1]
    width = lut_stack.shape[2]
    
    lut_batch = torch.zeros(size=(1, no_steps, height, width, 3), dtype=lut_stack.dtype, device='cpu') # shape: [1, 12, H, W, 3]
    lut_batch[0, :, :, :, :2] = lut_stack

    # feat_l has shape [N, 3, H, W]                
    feat_l_5D = img_left[:, :, None, :, :] # [N, 3, 1, H, W]
    feat_r_5D = img_right[:, :, None, :, :] # [N, 3, 1, H, W]

    feat_l_5D_rep = feat_l_5D.repeat([1, 1, no_steps, 1, 1])
    output_image_feat = torch.nn.functional.grid_sample(feat_r_5D, lut_batch, align_corners=False) # [1, 3, 12, H, W]

    image_right_transformed = torch.take_along_dim(output_image_feat[0], disp_index, axis=1)
    image_right_transformed = image_right_transformed.permute(1, 0, 2, 3)
    image_right_transformed[inv_mask_rgb] = float('nan')

    superposition = 0.5 * img_left + 0.5 * image_right_transformed
   
    if args.plot:
        fig, ax = plt.subplots(2,2)
        ax[0, 0].imshow(img_left[0].permute(1,2,0).detach().cpu())
        ax[0, 0].set_title("left image")
        ax[0, 1].imshow(img_right[0].permute(1,2,0).detach().cpu())
        ax[0, 1].set_title("right image")
        ax[1, 0].imshow(image_right_transformed[0].permute(1,2,0).detach().cpu())
        ax[1, 0].set_title("transformed right image")
        ax[1, 1].imshow(superposition[0].permute(1,2,0).detach().cpu())
        ax[1, 1].set_title("superposition: left and transformed right image")
        plt.show()

    if args.save_plots:
        save_tensor(image_right_transformed, "right_trans.webp")
        save_tensor(superposition, "superpos.webp")




