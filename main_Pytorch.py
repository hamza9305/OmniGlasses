import torch
import math
import timeit
import os
import matplotlib.pyplot as plt
from virtual_projection_Pytorch import Image_to_3D_Virtual_Space
from Euclidean_depth_map_pytorch import Depth_map_to_Euclidean_map
from Exr_Files_Pytorch import import_exr
from Euclidean_conversion_pytorch import Euclidean_conversion
import numpy as np
from torchvision import transforms
import torch.nn as nn
import argparse


def plot(map, title,  cmap_name="gist_rainbow", rad2deg=False):
    cmap = plt.get_cmap(cmap_name)
    if rad2deg:
        map = np.copy(map)
        map *= 180 / np.pi
    pos = plt.imshow(map, cmap)
    plt.title(title)
    plt.colorbar(pos)
    plt.show()\

def plot_GPU(map, title,  cmap_name="gist_rainbow", rad2deg=False,vmin=None, vmax= None, xlabel = None, svg = None,save_title = None):
    cmap = plt.get_cmap(cmap_name)
    if rad2deg:
        map = torch.clone(map)
        map *= 180 / np.pi
    map = map.to('cpu').detach()
    pos = plt.imshow(map, cmap)
    plt.title(title)
    plt.colorbar(pos)
    plt.clim(vmin,vmax)
    plt.xlabel(xlabel)
    if svg == True:
        plt.savefig(f"{save_title}.svg")

    plt.show()


def make_jitter(image, n=100):
    image_cpy = image.clone()
    image_cpy[::n, :, :] = 0
    image_cpy[:, ::n, :] = 0
    image_cpy[1::n, :, :] = 0
    image_cpy[:, 1::n, :] = 0
    return image_cpy

def compare_plot(output,left_image,right_image):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle('Vertically stacked subplots')

    output_cpy = output.clone().to('cpu').detach()
    left_image_cpy = left_image.clone().to('cpu').detach()
    right_image_cpy = right_image.clone().to('cpu').detach()

    output_vis = make_jitter(output_cpy, n=100)
    left_image_vis = make_jitter(left_image_cpy, n=100)
    right_image_vis = make_jitter(right_image_cpy, n=100)

    axs[0, 0].imshow(output_vis)
    axs[0, 0].set_title('output')

    axs[0, 1].imshow(right_image_vis)
    axs[0, 1].set_title('right_image')

    axs[1, 0].imshow(left_image_vis)
    axs[1, 0].set_title('left_image')

    axs[1, 1].imshow(right_image_vis)
    axs[1, 1].set_title('right_image')

    plt.show()




def main():

    parser = argparse.ArgumentParser(description='Generate Lookup tables')
    parser.add_argument('--path', default='masks_and_luts')
    parser.add_argument('--max_disp_deg', type=float, default=None, help="maximum allowed disparity in degrees (cannot be used with --max_disp_rad)")
    parser.add_argument('--max_disp_rad', type=float, default=None, help="maximum allowed disparity in radians [default: 0.31]")
    parser.add_argument('--plot', action='store_true', help="Plot debug plots")
    parser.add_argument('--save_plots', action='store_true', help="Save debug plots as svg files [needs --plot]")
    parser.add_argument('--num_stages', default=3, type=int, help="number of stages needing a lookup table")
    parser.add_argument('--device', default="cuda:0", type=str, help="tensor processing device, e.g. cuda:0")

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
    # path_x = path_stages + '/' + "lutx.pt"
    # path_y = path_stages + '/' + "luty.pt"
    path_mask = path_stages + '/' + "mask_stage%d.pt"
    path_beta_right = path_stages + '/' + "beta_rightstage%d.pt"
    path_light_ray = path_stages + '/' + "light_raystage%d.pt"
    path_lut_stack = path_stages + '/' + "lut_stage%d.pt"

    device = torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    convert_tensor = transforms.ToTensor()

    #baseline
    baseline = 0.30

    #Field of view
    FOV = math.pi

    #Allowed field of view
    allowed_theta = FOV / 2

    #step size
    # no_steps = 192 # 192
    upscaling_factors = [1, 2, 4, 16]
    no_steps_all_stages = [12, 25, 51, 201]
    max_disp_index_all_stages = [i-1 for i in no_steps_all_stages]

    stage_4_w = 1024
    stage_4_h = 1024

    width_all_stages = [int(stage_4_w / 16 * upscaling_factors[i]) for i in range(4)]
    height_all_stages = [int(stage_4_h / 16 * upscaling_factors[i]) for i in range(4)]

    #initialisation optical axis
    optical_axis = torch.tensor([0, 0, 1], device=device, dtype=torch.float)
    optical_axis = optical_axis.reshape([3, 1])

    for stage in range(args.num_stages):
        denominator = max_disp_index_all_stages[0] * upscaling_factors[stage]
        step_size = max_disp_rad / denominator
        no_steps = no_steps_all_stages[stage]
        max_disp_index = max_disp_index_all_stages[stage]
        width = width_all_stages[stage]
        height = height_all_stages[stage]

        #Image centre
        cx = (height - 1)/2
        cy = (width - 1)/2

        # #Meshgrid initialisation
        range_height = torch.arange(0,height).to(device)
        range_width = torch.arange(0, width).to(device)

        grid_x, grid_y = torch.meshgrid([range_height, range_width],indexing='xy')

        cur_path_lut_stack = path_lut_stack % (stage, stage)
        cur_path_mask = path_mask % (stage, stage)
        cur_path_beta_right = path_beta_right % (stage, stage)
        cur_path_light_ray = path_light_ray % (stage, stage)

        if os.path.exists(cur_path_lut_stack) and os.path.exists(cur_path_mask) \
                and os.path.exists(cur_path_beta_right) and os.path.exists(cur_path_light_ray):
            try:
                # lut_x = torch.load(path_x % stage, map_location=device)
                # lut_y = torch.load(path_y % stage, map_location=device)
                lut_stack = torch.load(cur_path_lut_stack, map_location=device)
                mask = torch.load(cur_path_mask, map_location=device)
                beta_right = torch.load(cur_path_beta_right, map_location=device)
                lightray = torch.load(cur_path_light_ray, map_location=device)

            except FileNotFoundError:
                print('File does not exist')
                exit(1)
        else:
            lut_x = torch.zeros(height, width, no_steps, device=device)
            lut_y = torch.zeros(height, width, no_steps, device=device)
            beta_right = torch.zeros(height, width, no_steps, device=device)
            mask = torch.zeros(height, width, no_steps, dtype=bool, device=device)


            angles = Image_to_3D_Virtual_Space(grid_x, grid_y, cx, cy, width, height, FOV, optical_axis)

            # finding the points from center
            cen_x, cen_y = angles.points_from_centre(angles.x, angles.y)
            print(f'The points from centre are {cen_x, cen_x}')
            if args.plot:
                plot_GPU(cen_x,"centre_x", svg=args.save_plots, save_title='cen_x')
                plot_GPU(cen_y, "centre_y", svg=args.save_plots, save_title='cen_y')


            length_vec = angles.length(cen_x, cen_y)
            print(f'The length of the Ray is {length_vec}')
            if args.plot:
                plot_GPU(length_vec, title = 'Length of the ray',svg=args.save_plots, save_title='length_vec')


            # Angle phi
            angle_phi = angles.phi(cen_x, cen_y)
            print(f'The angle phi is {angle_phi}')
            if args.plot:
                plot_GPU(angle_phi, title = 'Angle Phi',svg=args.save_plots, save_title='angle_phi')

            focal_length = angles.focal_length(angles.width)
            print(f'The focal length is {focal_length}')

            angle_theta = angles.theta(length_vec, focal_length)
            print(f'The angle theta is {angle_theta}')

            if args.plot:
                plot_GPU(angle_theta,'angle_theta',svg=args.save_plots, save_title='angle_theta')
            angle_theta_reshaped = angle_theta.reshape(angles.width, angles.height)
            mask1 = angle_theta_reshaped > allowed_theta


            # light_ray
            light_ray = angles.light_ray(angle_phi, angle_theta)
            lightray = torch.transpose(light_ray,0,1)
            lightray = torch.reshape(lightray,(height,width,3))
            # for plotting purposes
            if args.plot:
                plot_GPU(lightray[:, :, 0], 'lightray_x', svg=args.save_plots, save_title='lightray_x')
                plot_GPU(lightray[:, :, 0], 'lightray_y', svg=args.save_plots, save_title='lightray_y')
                plot_GPU(lightray[:, :, 0], 'lightray_z', svg=args.save_plots, save_title='lightray_z')
                plot_GPU(lightray, 'lightray', svg=args.save_plots, save_title='lightray')


            mask_cur_res_one_dim = angle_theta_reshaped > allowed_theta
            mask_cur_res = mask_cur_res_one_dim[:,:, None]
            mask_cur_res = torch.cat((mask_cur_res,) * 3, axis=2)
            
            ###################### for light ray
            # mask2 = lightray < 0
            # mask_light = torch.logical_or(mask2, mask_full_res)

            lightray[mask_cur_res] = 0
            # plot_GPU(lightray, 'lightray2')
            # path_light = '/home/haahm/PycharmProjects/Master_Thesis/FisheyeNet_MultipleLuts/AnyNet/Lookup_tables/Light_ray/lightray.pt'
            # torch.save(lightray.contiguous(), path_light)
            # exit()
            ##########################


            # q_ray
            q_ray = angles.vector_q(light_ray, device)
            print(q_ray.shape)
            q_ray_reshape = torch.reshape(q_ray, (height, width, 3))
            if args.plot:
                plot_GPU(q_ray_reshape[:, :, 0], 'q_ray', svg=args.save_plots, save_title='q_ray')
                plot_GPU(q_ray_reshape[:, :, 1], 'q_ray', svg=args.save_plots, save_title='q_ray')
                plot_GPU(q_ray_reshape[:, :, 2], 'q_ray', svg=args.save_plots, save_title='q_ray')
            # print(f'The vector q is {q_ray}')


            # angle alpha
            angle_alpha = angles.alpha(q_ray)
            print(f'The angle alpha is {angle_alpha}')
            if args.plot:
                plot_GPU(angle_alpha,'angle_alpha', svg=args.save_plots, save_title='angle_alpha')

            # angle_beta_left
            angle_beta_left = angles.beta_left(light_ray)
            print(f'The angle beta_left is {angle_beta_left.size()}')
            angle_beta_left_reshaped = angle_beta_left.reshape((height,width))
            angle_beta_left_reshaped[mask_cur_res_one_dim] = 0
            if args.plot:
                plot_GPU(angle_beta_left_reshaped,'angle_beta_left', svg=args.save_plots, save_title='angle_beta_left')


            start = timeit.default_timer()


            # denominator is 11, 22, 44
            print(f'{denominator=}')

            max_disp_rad_search = max_disp_rad / denominator * max_disp_index
            print(f'{max_disp_rad_search=}')


            #for index, disp in enumerate(np.arange(0, max_disp_rad_search + 1e-6, step_size)):
            for index, disp in enumerate(np.linspace(0, max_disp_rad_search, no_steps)):

                angle_beta_right = angles.beta_right(angle_beta_left, disp)
                print(f'The angle beta_right is {angle_beta_right}')
                beta_mask = torch.tensor(angle_beta_right < 0)
                angle_beta_right[beta_mask] = torch.inf
                beta_right[:, :, index] = angle_beta_right.reshape(angles.height, angles.width)

                light_ray_right = angles.light_ray_right(angle_beta_right, angle_alpha)
                print(f'The light ray right is {light_ray_right}')

                light_ray_right_reshaped = torch.reshape(light_ray_right[2],(height,width))
                # mask2 = light_ray_right_reshaped < 0

                try:
                    # mask[:, :, index] = torch.logical_or(mask2, mask1)
                    mask[:, :, index] = mask1
                except Exception as e:
                    raise

                light_ray_right_without_z = angles.light_ray_right_without_z(light_ray_right)
                print(f'The light ray right without z_coords {light_ray_right_without_z}')

                angle_theta_right = angles.angle_theta_right(light_ray_right)
                print(f'The angle theta of light ray right is {angle_theta_right}')
                angle_theta_right_reshaped = angle_theta_right.reshape(angles.width, angles.height)

                length_right_ray = angles.length_right_ray(angle_theta_right, focal_length)
                print(f'The length of light ray right is {length_right_ray}')
                length_right_ray_reshaped = length_right_ray.reshape(angles.height, angles.width)

                coord_x, coord_y = angles.coordinates(light_ray_right_without_z, length_right_ray)
                print(f'The coordinates are {coord_x}')

                lut_x[:, :, index] = coord_x.reshape(angles.width, angles.height)
                lut_y[:, :, index] = coord_y.reshape(angles.width, angles.height)

            #lookup tables rearranging
            lut_x = -1 + (1 / width) + lut_x * (2 / width)
            lut_y = -1 + (1 / height) + lut_y * (2 / height)

            lut_x = torch.unsqueeze(lut_x,0)
            lut_y = torch.unsqueeze(lut_y, 0)

            lut_stack = torch.stack([lut_x,lut_y], dim = 3)
            lut_stack = lut_stack.permute(4,0,1,2,3)

            stop = timeit.default_timer()
            print('Time to build lookup table', stop - start)
            # torch.save(lut_x.contiguous(), path_x)
            # torch.save(lut_y.contiguous(), path_y)

            #uncomment
            print(f'{path_lut_stack=}')

            os.makedirs(os.path.dirname(cur_path_lut_stack), exist_ok=True)

            torch.save(lut_stack.contiguous(), cur_path_lut_stack)
            torch.save(mask.permute(2, 0, 1).contiguous(), cur_path_mask)
            # torch.save(beta_right.contiguous(), cur_path_beta_right) # for depth to disparity conv
            # torch.save(lightray.contiguous(), cur_path_light_ray) # for depth to disparity conv


    # MASK FULL RESOLUTION

    width = stage_4_w
    height = stage_4_h

    #Image centre
    cx = (height - 1)/2
    cy = (width - 1)/2

    # #Meshgrid initialisation
    range_height = torch.arange(0,height).to(device)
    range_width = torch.arange(0, width).to(device)

    grid_x, grid_y = torch.meshgrid([range_height, range_width],indexing='xy')
    angles = Image_to_3D_Virtual_Space(grid_x, grid_y, cx, cy, width, height, FOV, optical_axis)

    # finding the points from center
    cen_x, cen_y = angles.points_from_centre(angles.x, angles.y)
    print(f'The points from centre are {cen_x, cen_x}')
    
    focal_length = angles.focal_length(angles.width)
    print(f'The focal length is {focal_length}')

    length_vec = angles.length(cen_x, cen_y)
    print(f'The length of the Ray is {length_vec}')

    angle_theta = angles.theta(length_vec, focal_length)
    print(f'The angle theta is {angle_theta}')
    angle_theta_reshaped = angle_theta.reshape(angles.width, angles.height)
    
    mask_full_res_one_dim = angle_theta_reshaped > allowed_theta
    if args.plot:
        plot_GPU(mask_full_res_one_dim,'mask_full_res 1D',svg=args.save_plots, save_title='mask_full_res_1D')

    mask_full_res = mask_full_res_one_dim[None, :, :] # [1, H, W]
    path_mask_new = os.path.join(args.path, 'mask_full_res.pt')
    os.makedirs(args.path, exist_ok=True)
    torch.save(mask_full_res.contiguous(), path_mask_new)


if __name__ == '__main__':
    main()
