import numpy as np
import torch


def Depth_map_to_Euclidean_map(depth_map, projection_light):

    projection_light_z =  projection_light[:, :, 2]
    euclidean_depth_image = depth_map/projection_light_z


    return torch.unsqueeze(euclidean_depth_image, dim = -1)


