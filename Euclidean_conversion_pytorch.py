import torch


def Euclidean_conversion(baseline, beta_right, disparity):

    disparity_cpy  = disparity.clone()
    invalid_mask_a = disparity_cpy<= 0
    invalid_mask_b = disparity_cpy != disparity_cpy
    invalid_mask_c = disparity_cpy == torch.inf

    invalid_mask_ab = torch.bitwise_or(invalid_mask_a, invalid_mask_b)
    invalid_mask = torch.bitwise_or(invalid_mask_ab, invalid_mask_c)
    disparity_cpy[invalid_mask] = 1  # to avoid division by 0

    euclidean = baseline * torch.sin(beta_right) / torch.sin(disparity_cpy)

    return euclidean